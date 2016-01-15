LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_LDLIBS    := -lm -llog 
include ../OpenCV-android-sdk/native/jni/OpenCV.mk
LOCAL_SRC_FILES  := ImageProc.cpp
LOCAL_MODULE     := image_proc
include $(BUILD_SHARED_LIBRARY)