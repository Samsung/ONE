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

#ifndef __NNFW_INTERNAL_H__
#define __NNFW_INTERNAL_H__

#include "nnfw.h"

NNFW_STATUS nnfw_set_config(nnfw_session *session, const char *key, const char *value);

NNFW_STATUS nnfw_get_config(nnfw_session *session, const char *key, char *value, size_t value_size);

/**
 * @brief Load a circle model from buffer.
 *
 * The buffer must outlive the session.
 *
 * @param[in] session session
 * @param[in] buffer  Pointer to the buffer
 * @param[in] size    Buffer size
 * @return NNFW_STATUS
 */
NNFW_STATUS nnfw_load_circle_from_buffer(nnfw_session *session, uint8_t *buffer, size_t size);

/**
 * @brief Load a tflite/circle model from file.
 *
 * @param[in] session   session
 * @param[in] file_path Path to model file. Model type(tflite/circle) is decided by file extension
 * @return    NFNFW_STATUS
 */
NNFW_STATUS nnfw_load_model_from_modelfile(nnfw_session *session, const char *file_path);

#endif // __NNFW_INTERNAL_H__
