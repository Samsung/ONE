LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

ifndef ONERT_PREBUILT_LIB_DIR
$(error ONERT_PREBUILT_LIB_DIR is not set)
endif

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
		$(ONERT_PREBUILT_LIB_DIR)/nnfw/backend/libbackend_cpu.so
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
