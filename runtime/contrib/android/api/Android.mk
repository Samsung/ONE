LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

API_ROOT_PATH := $(LOCAL_PATH)
PREBUILT_LIB :=

include $(API_ROOT_PATH)/prebuilt/Android.mk
include $(API_ROOT_PATH)/src/main/native/Android.mk

#$(warning $(PREBUILT_LIB))
