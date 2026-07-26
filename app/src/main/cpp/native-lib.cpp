#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_drgnflyengine_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_drgnflyengine_MainActivity_processFrames(JNIEnv *env, jobject thiz,
                                                          jobject pixel_data, jobject overlay_bitmap, jint width,
                                                          jint height) {

    //Pointer to the address where pixels are dumped by the camera
    auto* pixel_buffer_address = static_cast<uint8_t *>(env->GetDirectBufferAddress(pixel_data));
    Mat raw_gray(height, width, CV_8UC1, pixel_buffer_address);
    Mat frame = raw_gray.clone();

    //rotate(raw_gray, raw_gray, ROTATE_90_CLOCKWISE);

    //temp log to make sure pixel data is reaching here
    __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++", "Image width: %d, Image height: %d", width, height);

    // Testing Java UI overlay with canny edge detection
    Mat blur_img, canny_img;
    GaussianBlur(frame, blur_img, Size(5, 5), 0, 0);
    Canny(blur_img, canny_img, 50, 150);

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, overlay_bitmap, &info);


    // If Java hands us the wrong sized glass, abort immediately to prevent a SIGSEGV
    if (info.width != width || info.height != height) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++",
                            "CRASH AVERTED: Camera is %dx%d but Java Bitmap is %dx%d!",
                            height, width, info.height, info.width);
        return;
    }

    // Locking Bitmap in RAM
    void* bitmap_pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, overlay_bitmap, &bitmap_pixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "DRGNFLY_C++", "Failed to lock Bitmap pixels.");
        return;
    }

    // Wiping and writing on the glass (overlay)
    Mat overlay_mat(height, width, CV_8UC4, bitmap_pixels);
    overlay_mat.setTo(Scalar(0, 0, 0, 0));
    overlay_mat.setTo(Scalar(0, 255, 0, 255), canny_img);

    AndroidBitmap_unlockPixels(env, overlay_bitmap);

    // TODO: implement processFrames() ML logic
}