/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_DEBUG_H__
#define __NNFW_DEBUG_H__

#include "nnfw.h"

/**
 * @brief Extension of NNFW_INFO_ID for API in nnfw_debug.h
 */
typedef enum {

  /* Experimental. Subject to be changed without notice. */
  NNFW_INFO_BACKENDS,

  /* Experimental. Subject to be changed without notice. */
  NNFW_INFO_EXECUTOR,

} NNFW_INFO_ID_EX;

NNFW_STATUS nnfw_create_debug_session(nnfw_session **session);

NNFW_STATUS nnfw_set_config(nnfw_session *session, const char *key, const char *value);

NNFW_STATUS nnfw_query_info_str(nnfw_session *session, NNFW_INFO_ID_EX id, char *value);

#endif // __NNFW_DEBUG_H__
