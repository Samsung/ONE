LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-dev
PREBUILT_LIB += nnfw-dev
LOCAL_SRC_FILES := \
		libnnfw-dev.so
include $(PREBUILT_SHARED_LIBRARY)
