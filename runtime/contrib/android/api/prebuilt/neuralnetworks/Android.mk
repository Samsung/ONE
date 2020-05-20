LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := neuralnetworks
PREBUILT_LIB += neuralnetworks
LOCAL_SRC_FILES := \
		libneuralnetworks.so
include $(PREBUILT_SHARED_LIBRARY)
