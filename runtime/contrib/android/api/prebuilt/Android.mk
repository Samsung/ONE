LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
PREBUILT_PATH := $(LOCAL_PATH)
include $(PREBUILT_PATH)/backend_cpu/Android.mk
include $(PREBUILT_PATH)/circle_loader/Android.mk
include $(PREBUILT_PATH)/neuralnetworks/Android.mk
include $(PREBUILT_PATH)/nnfw-dev/Android.mk
include $(PREBUILT_PATH)/onert_core/Android.mk
include $(PREBUILT_PATH)/tensorflowlite_jni/Android.mk
include $(PREBUILT_PATH)/tflite_loader/Android.mk
