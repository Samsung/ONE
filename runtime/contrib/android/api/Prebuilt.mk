LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifndef ONERT_PREBUILT_LIB_DIR
$(error ONERT_PREBUILT_LIB_DIR is not set)
endif

# libcircle_loader
include $(CLEAR_VARS)
LOCAL_MODULE := circle_loader
PREBUILT_LIB += circle_loader
LOCAL_SRC_FILES := \
		$(ONERT_PREBUILT_LIB_DIR)/libcircle_loader.so
include $(PREBUILT_SHARED_LIBRARY)

# libtflite_loader
include $(CLEAR_VARS)
LOCAL_MODULE := tflite_loader
PREBUILT_LIB += tflite_loader
LOCAL_SRC_FILES := \
		$(ONERT_PREBUILT_LIB_DIR)/libtflite_loader.so
include $(PREBUILT_SHARED_LIBRARY)

# libtensorflowlite_jni
include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite_jni
PREBUILT_LIB += tensorflowlite_jni
LOCAL_SRC_FILES := \
		$(ONERT_PREBUILT_LIB_DIR)/libtensorflowlite_jni.so
include $(PREBUILT_SHARED_LIBRARY)

# libnnfw
include $(CLEAR_VARS)
LOCAL_MODULE := nnfw-dev
PREBUILT_LIB += nnfw-dev
LOCAL_SRC_FILES := \
		$(ONERT_PREBUILT_LIB_DIR)/libnnfw-dev.so
include $(PREBUILT_SHARED_LIBRARY)

# libonert_core
include $(CLEAR_VARS)
LOCAL_MODULE := onert_core
PREBUILT_LIB += onert_core
LOCAL_SRC_FILES := \
		$(ONERT_PREBUILT_LIB_DIR)/libonert_core.so
include $(PREBUILT_SHARED_LIBRARY)

# backend_cpu
include $(CLEAR_VARS)
LOCAL_MODULE := backend_cpu
PREBUILT_LIB += backend_cpu
LOCAL_SRC_FILES := \
		$(ONERT_PREBUILT_LIB_DIR)/libbackend_cpu.so
include $(PREBUILT_SHARED_LIBRARY)

# TODO Support backend acl
# backend_acl
ifeq ($(ONERT_CONTAINS_ACL), 1)
	$(error containing acl backend doesn't supported yet)
endif

# backend_ext
ifneq ($(ONERT_EXT_PREBUILT_LIB), )
include $(CLEAR_VARS)
LOCAL_MODULE := backend_ext
PREBUILT_LIB += backend_ext
LOCAL_SRC_FILES := \
		$(ONERT_EXT_PREBUILT_LIB)
include $(PREBUILT_SHARED_LIBRARY)
endif
