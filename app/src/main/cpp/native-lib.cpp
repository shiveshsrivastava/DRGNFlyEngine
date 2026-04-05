#include <jni.h>
#include <string>
#include <android/log.h>

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
                                                          jobject pixel_data, jint width,
                                                          jint height) {

    //Pointer to the address where pixels are dumped by the camera
    auto* pixel_buffer_address = static_cast<uint8_t *>(env->GetDirectBufferAddress(pixel_data));

    //temp log to make sure pixel data is reaching here
    __android_log_print(ANDROID_LOG_INFO, "DRGNFLY_C++", "Image width: %d, Image height: %d", width, height);
    // TODO: implement processFrames() logic
}