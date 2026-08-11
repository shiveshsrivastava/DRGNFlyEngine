#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <optional>
#include <vector>
#include <chrono>
#include <atomic>
#include "litert/cc/litert_buffer_ref.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "litert/cc/litert_macros.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

using namespace cv;

static std::optional<litert::Environment> g_env;

static std::optional<litert::CompiledModel> g_compiled_model;
static std::vector<litert::TensorBuffer> g_input_buffers;
static std::vector<litert::TensorBuffer> g_output_buffers;

static std::atomic<bool> g_init_started{false};

// Deliberately never closed: it backs the model buffer for the process lifetime.
static AAsset* g_model_asset = nullptr;

// Fraction of frame height fed to the model. 0.267 is distortion-free (5:1);
// larger values buy field of view at the cost of vertical squash.
static constexpr float kCropFraction = 0.40f;



static constexpr const char* kModelAsset = "culane_res18_pixel9p.tflite";

// UFLDv2 CULane has two heads. Row anchors answer "at this image row, where is the
// lane?", which is well posed for the near-vertical ego lanes. Column anchors answer
// the transposed question, which is the well posed one for the shallow outer lanes.
// The reference decode picks the head by lane index, so we do the same.
static constexpr int kGridRow = 200, kAnchorsRow = 72;
static constexpr int kGridCol = 100, kAnchorsCol = 81;
static constexpr int kLanes = 4, kWin = 1;
static constexpr bool kUseGpu = true;

// Anchors are fractions of the ORIGINAL image, and the network input is its bottom
// 60%. Converts an original-image fraction to a fraction of the input.
static constexpr float kTrainCrop = 0.6f;
static inline float ToInputFraction(float original_fraction) {
    return (original_fraction - (1.0f - kTrainCrop)) / kTrainCrop;
}

// Argmax over the grid, then a softmax expectation over a window around it.
// `base` points at grid cell 0; consecutive cells are `stride` floats apart.
// Rejects anchors whose distribution is too flat to locate a lane.
static bool GridExpectation(const float* base, int num_grid, int stride,
                            float min_confidence, float* out) {
    int best = 0;
    float bv = -1e30f;
    for (int g = 0; g < num_grid; ++g) {
        const float v = base[g * stride];
        if (v > bv) { bv = v; best = g; }
    }
    // Full-grid softmax denominator, so confidence is the peak's share of the
    // whole distribution rather than just of the local window.
    float total = 0.f;
    for (int g = 0; g < num_grid; ++g) total += std::exp(base[g * stride] - bv);
    if (total <= 0.f || 1.0f / total < min_confidence) return false;

    const int lo = std::max(0, best - kWin);
    const int hi = std::min(num_grid - 1, best + kWin);
    float wsum = 0.f, gsum = 0.f;
    for (int g = lo; g <= hi; ++g) {
        const float w = std::exp(base[g * stride] - bv);
        wsum += w;
        gsum += w * static_cast<float>(g);
    }
    if (wsum <= 0.f) return false;
    *out = gsum / wsum;
    return true;
}


static int CountValid(const float* exist, int num_anchors, int lane) {
    int n = 0;
    for (int a = 0; a < num_anchors; ++a) {
        const int e = a * kLanes + lane;
        if (exist[num_anchors * kLanes + e] > exist[e]) ++n;
    }
    return n;
}



static void LogBuffers(const char* label,
                       const std::vector<litert::TensorBuffer>& buffers) {
    for (size_t i = 0; i < buffers.size(); ++i) {
        auto type = buffers[i].TensorType();
        if (!type) continue;
        std::string dims;
        for (auto d : type->Layout().Dimensions()) {
            if (!dims.empty()) dims += ", ";
            dims += std::to_string(d);
        }
        auto size = buffers[i].Size();
        auto packed = buffers[i].PackedSize();
        __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                            "%s[%zu] dims=[%s] size=%zu packed=%zu",
                            label, i, dims.c_str(),
                            size ? *size : 0u, packed ? *packed : 0u);
    }
}

static void BenchmarkRun(int iterations) {
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto ok = g_compiled_model->Run(g_input_buffers, g_output_buffers);
        const auto t1 = std::chrono::steady_clock::now();
        if (!ok) {
            __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++",
                                "Run failed: %s", ok.Error().Message().c_str());
            return;
        }
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                            "Run #%d: %.1f ms%s", i, ms,
                            i == 0 ? "  (cold: includes faulting in weights)" : "");
    }
}

// Allocate plain host-memory buffers instead of whatever the accelerator asks for.
static litert::Expected<std::vector<litert::TensorBuffer>> MakeHostBuffers(bool is_input) {
    std::vector<litert::TensorBuffer> buffers;
    LITERT_ASSIGN_OR_RETURN(
        auto names, is_input ? g_compiled_model->GetSignatureInputNames()
                             : g_compiled_model->GetSignatureOutputNames());
    for (size_t i = 0; i < names.size(); ++i) {
        LITERT_ASSIGN_OR_RETURN(
            auto type, is_input ? g_compiled_model->GetInputTensorType(0, i)
                                : g_compiled_model->GetOutputTensorType(0, i));
        LITERT_ASSIGN_OR_RETURN(const size_t bytes, type.Bytes());
        LITERT_ASSIGN_OR_RETURN(
            auto buffer,
            litert::TensorBuffer::CreateManaged(
                *g_env, litert::TensorBufferType::kHostMemory, type, bytes));
        buffers.push_back(std::move(buffer));
    }
    __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                        "Allocated %zu host-memory %s buffers",
                        buffers.size(), is_input ? "input" : "output");
    return buffers;
}


static litert::Expected<void> InitEngine(AAssetManager* asset_manager) {
    LITERT_ASSIGN_OR_RETURN(auto litert_env, litert::Environment::Create({}));
    g_env = std::move(litert_env);

    g_model_asset = AAssetManager_open(asset_manager, kModelAsset, AASSET_MODE_BUFFER);
    if (g_model_asset == nullptr) {
        return litert::Error(litert::Status::kErrorNotFound,
                             std::string("Asset not found: ") + kModelAsset);
    }
    const void* data = AAsset_getBuffer(g_model_asset);
    const off_t length = AAsset_getLength(g_model_asset);
    if (data == nullptr) {
        return litert::Error(litert::Status::kErrorRuntimeFailure,
                             "AAsset_getBuffer returned null");
    }
    __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                        "Model mapped: %lld bytes", (long long)length);

        LITERT_ASSIGN_OR_RETURN(auto options, litert::Options::Create());
    options.SetHardwareAccelerators(kUseGpu ? litert::HwAccelerators::kGpu : litert::HwAccelerators::kCpu);

    // The model is dynamic-range quantized (int8 weights, fp32 activations).
    // Without this the GPU accelerator rejects the quantized conv and fc ops
    // and hands most of the graph back to the CPU.
    auto gpu_opts = options.GetGpuOptions();
    if (gpu_opts) {
        gpu_opts->EnableAllowSrcQuantizedFcConvOps(true);
        gpu_opts->SetBackend(litert::GpuOptions::Backend::kOpenCl);
        gpu_opts->HintWaitingForCompletion(true);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "DRGNFLY_C++",
                            "GetGpuOptions failed: %s",
                            gpu_opts.Error().Message().c_str());
    }

    LITERT_ASSIGN_OR_RETURN(auto compiled,
        litert::CompiledModel::Create(
            *g_env, litert::BufferRef<uint8_t>(data, static_cast<size_t>(length)),
            options));

    g_compiled_model = std::move(compiled);

    auto in_bufs = g_compiled_model->CreateInputBuffers();
    if (in_bufs) {
        g_input_buffers = std::move(*in_bufs);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "DRGNFLY_C++",
                            "CreateInputBuffers failed (%s); using host memory",
                            in_bufs.Error().Message().c_str());
        LITERT_ASSIGN_OR_RETURN(g_input_buffers, MakeHostBuffers(true));
    }

    auto out_bufs = g_compiled_model->CreateOutputBuffers();
    if (out_bufs) {
        g_output_buffers = std::move(*out_bufs);
    } else {
        __android_log_print(ANDROID_LOG_WARN, "DRGNFLY_C++",
                            "CreateOutputBuffers failed (%s); using host memory",
                            out_bufs.Error().Message().c_str());
        LITERT_ASSIGN_OR_RETURN(g_output_buffers, MakeHostBuffers(false));
    }



    LogBuffers("input", g_input_buffers);
    LogBuffers("output", g_output_buffers);
    //BenchmarkRun(5);
    return {};
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_drgnflyengine_MainActivity_initModel(JNIEnv *env, jobject thiz,
                                                      jobject asset_manager_obj) {

     if (g_init_started.exchange(true)) {
        __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                            "Model already initialised; skipping.");
        return g_compiled_model.has_value() ? JNI_TRUE : JNI_FALSE;
    }


    AAssetManager* asset_manager = AAssetManager_fromJava(env, asset_manager_obj);
    auto result = InitEngine(asset_manager);
    if (!result) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++",
                            "LiteRT init failed: %s",
                            result.Error().Message().c_str());
        return JNI_FALSE;
    }
    __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                        "LiteRT model ready.");
    return JNI_TRUE;
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_drgnflyengine_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

// Maps uint8 -> (v/255 - mean) / std, per channel, in a single pass.
static Mat MakeNormalizeLut() {
    Mat lut(1, 256, CV_32FC3);
    const float mean[3]  = {0.485f, 0.456f, 0.406f};
    const float stdev[3] = {0.229f, 0.224f, 0.225f};
    for (int v = 0; v < 256; ++v) {
        Vec3f& p = lut.at<Vec3f>(0, v);
        for (int c = 0; c < 3; ++c) {
            p[c] = (v / 255.0f - mean[c]) / stdev[c];
        }
    }
    return lut;
}


static void RunInference(const Mat& model_input) {
    auto in_ptr = g_input_buffers[0].Lock(litert::TensorBuffer::LockMode::kWrite);
    if (!in_ptr) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++", "input Lock failed: %s",
                            in_ptr.Error().Message().c_str());
        return;
    }

    auto sz = g_input_buffers[0].Size();
    auto pk = g_input_buffers[0].PackedSize();
    if (!sz || !pk || *sz != *pk || *sz != 320u * 1600u * 3u * sizeof(float)) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++",
                            "input buffer not densely packed: size=%zu packed=%zu",
                            sz ? *sz : 0u, pk ? *pk : 0u);
        g_input_buffers[0].Unlock();
        return;
    }

    static const Mat kNormalizeLut = MakeNormalizeLut();
    Mat tensor_view(320, 1600, CV_32FC3, *in_ptr);
    LUT(model_input, kNormalizeLut, tensor_view);
    g_input_buffers[0].Unlock();

    auto ok = g_compiled_model->Run(g_input_buffers, g_output_buffers);
    if (!ok) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++", "Run failed: %s",
                            ok.Error().Message().c_str());
        return;
    }
}

static void DrawLanes(Mat& overlay, int y_start, float sx, float sy) {

    if (g_output_buffers.size() < 4) 
        return;

    auto lc = g_output_buffers[0].Lock(litert::TensorBuffer::LockMode::kRead);
    auto ec = g_output_buffers[1].Lock(litert::TensorBuffer::LockMode::kRead);
    auto lr = g_output_buffers[2].Lock(litert::TensorBuffer::LockMode::kRead);
    auto er = g_output_buffers[3].Lock(litert::TensorBuffer::LockMode::kRead);
    if (!lc || !ec || !lr || !er) {
        if (lc) g_output_buffers[0].Unlock();
        if (ec) g_output_buffers[1].Unlock();
        if (lr) g_output_buffers[2].Unlock();
        if (er) g_output_buffers[3].Unlock();
        return;
    }

    const float* LC = static_cast<const float*>(*lc);
    const float* EC = static_cast<const float*>(*ec);
    const float* LR = static_cast<const float*>(*lr);
    const float* ER = static_cast<const float*>(*er);

    int valid[kLanes];
    for (int l : {1, 2}) valid[l] = CountValid(ER, kAnchorsRow, l);
    for (int l : {0, 3}) valid[l] = CountValid(EC, kAnchorsCol, l);


    const Scalar colors[kLanes] = {
        Scalar(255, 64, 64, 255), Scalar(0, 255, 0, 255),
        Scalar(0, 200, 255, 255), Scalar(255, 0, 255, 255)};

    std::vector<Point> pts[kLanes];

    // Ego lanes from the row head. loc_row is [1,200,72,4], exist_row is [1,2,72,4].
    constexpr int kRowStride = kAnchorsRow * kLanes;
    for (int l : {1, 2}) {
        if (valid[l] * 2 <= kAnchorsRow) continue;
        for (int a = 0; a < kAnchorsRow; ++a) {
            const int e = a * kLanes + l;
            if (ER[kRowStride + e] <= ER[e]) continue;

            float gbar;
            if (!GridExpectation(LR + e, kGridRow, kRowStride, 0.10f, &gbar)) continue;

            const float xm = gbar / (kGridRow - 1) * 1600.0f;
            const float ym = ToInputFraction(
                0.42f + 0.58f * a / (kAnchorsRow - 1)) * 320.0f;
            pts[l].emplace_back(cvRound(xm * sx), cvRound(y_start + ym * sy));
        }
    }

    // Outer lanes from the column head. loc_col is [1,100,81,4], exist_col [1,2,81,4].
    // The 81 anchors are evenly spaced across the full width, and the grid predicts a
    // y position as a fraction of the original image height.
    constexpr int kColStride = kAnchorsCol * kLanes;
    for (int l : {0, 3}) {
        if (valid[l] * 4 <= kAnchorsCol) continue;
        for (int k = 0; k < kAnchorsCol; ++k) {
            const int e = k * kLanes + l;
            if (EC[kColStride + e] <= EC[e]) continue;

            float gbar;
            if (!GridExpectation(LC + e, kGridCol, kColStride, 0.10f, &gbar)) continue;

            const float t = ToInputFraction(gbar / (kGridCol - 1));
            if (t < 0.0f || t > 1.0f) continue;  // lands above our crop

            const float xm = static_cast<float>(k) / (kAnchorsCol - 1) * 1600.0f;
            pts[l].emplace_back(cvRound(xm * sx), cvRound(y_start + t * 320.0f * sy));
        }
    }

    int drawn[kLanes];
    for (int l = 0; l < kLanes; ++l) {
        for (const Point& p : pts[l]) circle(overlay, p, 6, colors[l], -1);
        if (pts[l].size() > 1) {
            std::vector<std::vector<Point>> poly{pts[l]};
            polylines(overlay, poly, false, colors[l], 3);
        }
        drawn[l] = static_cast<int>(pts[l].size());
    }

    for (int i = 0; i < 4; ++i) g_output_buffers[i].Unlock();

    static int frame_counter = 0;
    if (++frame_counter % 30 == 0) {
        __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                            "valid/lane: %d %d %d %d   drew/lane: %d %d %d %d",
                            valid[0], valid[1], valid[2], valid[3],
                            drawn[0], drawn[1], drawn[2], drawn[3]);
    }

}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_drgnflyengine_MainActivity_processFrames(JNIEnv *env, jobject thiz,
                                                          jobject pixel_data,
                                                          jobject overlay_bitmap,
                                                          jint width,
                                                          jint height,
                                                          jint row_stride) {

    auto* pixels = static_cast<uint8_t*>(env->GetDirectBufferAddress(pixel_data));
    if (pixels == nullptr) return;

    const auto t0 = std::chrono::steady_clock::now();

    // CameraX gives us RGBA_8888 in one plane. row_stride may exceed width*4.
    Mat rgba(height, width, CV_8UC4, pixels, static_cast<size_t>(row_stride));

    const int crop_height = std::min(height, cvRound(height * kCropFraction));
    const int y_start = height - crop_height;

    if (y_start < 0 || width < 1600) return;

    Mat roi_resized;
    resize(rgba(Rect(0, y_start, width, crop_height)), roi_resized,
       Size(1600, 320), 0, 0, INTER_AREA);


    Mat model_input;
    cvtColor(roi_resized, model_input, COLOR_RGBA2RGB);


    const auto t1 = std::chrono::steady_clock::now();

    if (g_compiled_model.has_value() && 
            !g_input_buffers.empty() && 
            g_output_buffers.size() >= 4) {
        RunInference(model_input);
    }

    const auto t2 = std::chrono::steady_clock::now();

    // Show the model's actual view on the overlay.
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, overlay_bitmap, &info);
    if (info.width != static_cast<uint32_t>(width) ||
        info.height != static_cast<uint32_t>(height)) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++",
                            "Bitmap %dx%d != camera %dx%d",
                            info.width, info.height, width, height);
        return;
    }
    void* bitmap_pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, overlay_bitmap, &bitmap_pixels) < 0) return;

    Mat overlay(height, width, CV_8UC4, bitmap_pixels);
    overlay.setTo(Scalar(0, 0, 0, 0));
    Mat thumb;
    resize(model_input, thumb, Size(480, 96));
    Mat corner = overlay(Rect(width - 480 - 16, 16, 480, 96));
    cvtColor(thumb, corner, COLOR_RGB2RGBA);


        DrawLanes(overlay, y_start,
              static_cast<float>(width) / 1600.0f,
              static_cast<float>(crop_height) / 320.0f);


    AndroidBitmap_unlockPixels(env, overlay_bitmap);

    const auto t3 = std::chrono::steady_clock::now();
    static int prof_counter = 0;
    if (++prof_counter % 30 == 0) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++",
                            "prep %.1f  infer %.1f  draw %.1f  total %.1f ms",
                            ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t0, t3));
    }

}