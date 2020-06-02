LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite_jni
PREBUILT_LIB += tensorflowlite_jni
LOCAL_SRC_FILES := \
		libtensorflowlite_jni.so
include $(PREBUILT_SHARED_LIBRARY)
