LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := backend_cpu
PREBUILT_LIB += backend_cpu
LOCAL_SRC_FILES := \
		libbackend_cpu.so
include $(PREBUILT_SHARED_LIBRARY)
