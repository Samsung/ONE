LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := circle_loader
PREBUILT_LIB += circle_loader
LOCAL_SRC_FILES := \
		libcircle_loader.so
include $(PREBUILT_SHARED_LIBRARY)
