LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := onert_core
PREBUILT_LIB += onert_core
LOCAL_SRC_FILES := \
		libonert_core.so
include $(PREBUILT_SHARED_LIBRARY)
