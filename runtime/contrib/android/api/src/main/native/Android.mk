LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifndef ONERT_API_INC_DIR
$(error ONERT_API_INC_DIR is not set)
endif

LOCAL_MODULE := onert-native-api
LOCAL_SRC_FILES := \
		onert-native-internal.cpp \
		onert-native-helper.cpp \
		onert-native-api.cpp
LOCAL_C_INCLUDES := $(ONERT_API_INC_DIR)
LOCAL_CXXFLAGS := -std=c++14 -O3 -fPIC -frtti -fexceptions
LOCAL_SHARED_LIBRARIES := $(PREBUILT_LIB)
LOCAL_LDLIBS := -llog -landroid

include $(BUILD_SHARED_LIBRARY)
