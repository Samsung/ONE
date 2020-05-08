LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := tflite_loader
PREBUILT_LIB += tflite_loader
LOCAL_SRC_FILES := \
		libtflite_loader.so
include $(PREBUILT_SHARED_LIBRARY)
