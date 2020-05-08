LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := onert-native
LOCAL_SRC_FILES := onert-native-api.cc

include $(BUILD_SHARED_LIBRARY)
