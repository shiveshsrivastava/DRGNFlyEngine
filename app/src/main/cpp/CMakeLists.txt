# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("drgnflyengine")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV dependency management

include(FetchContent)
set(OPENCV_VERSION "4.12.0")

FetchContent_Declare(
    opencv_android_sdk
    URL "https://github.com/opencv/opencv/releases/download/${OPENCV_VERSION}/opencv-${OPENCV_VERSION}-android-sdk.zip"
)

FetchContent_MakeAvailable(opencv_android_sdk)

set(LITERT_CC_SDK "${CMAKE_CURRENT_SOURCE_DIR}/../../../litert_cc_sdk/litert_cc_sdk")
set(LITERT_SO "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libLiteRt.so")
if (NOT EXISTS "${LITERT_SO}")
    message(FATAL_ERROR
        "libLiteRt.so missing at ${LITERT_SO} -- extract jni/${ANDROID_ABI}/libLiteRt.so "
        "from the LiteRT AAR (dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.1.6/litert-2.1.6.aar)")
endif()
file(COPY "${LITERT_SO}" DESTINATION "${LITERT_CC_SDK}")

add_subdirectory("${LITERT_CC_SDK}" "${CMAKE_BINARY_DIR}/litert_cc_sdk")

set(OpenCV_DIR "${opencv_android_sdk_SOURCE_DIR}/sdk/native/jni")
find_package(OpenCV REQUIRED)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
#
# In this top level CMakeLists.txt, ${CMAKE_PROJECT_NAME} is used to define
# the target library name; in the sub-module's CMakeLists.txt, ${PROJECT_NAME}
# is preferred for the same purpose.
#
# In order to load a library into your app from Java/Kotlin, you must call
# System.loadLibrary() and pass the name of the library defined here;
# for GameActivity/NativeActivity derived applications, the same library name must be
# used in the AndroidManifest.xml file.
add_library(${CMAKE_PROJECT_NAME} SHARED
    # List C/C++ source files with relative paths to this CMakeLists.txt.
    native-lib.cpp)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)

# Specifies libraries CMake should link to your target library. You
# can link libraries from various origins, such as libraries defined in this
# build script, prebuilt third-party libraries, or Android system libraries.
target_link_libraries(${CMAKE_PROJECT_NAME}
    # List libraries link to the target library
    android
    log
    ${OpenCV_LIBS}
    jnigraphics
    litert_cc_api)