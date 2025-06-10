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
 * @brief Export circle+ model
 * @note  This function should be called on training mode
 *        This function should be called before or after {@link nnfw_train}
 *
 * @param[in] session     The session to export training model
 * @param[in] file_path   The path to export training model
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_export_circleplus(nnfw_session *session, const char *file_path);

/**
 * @brief Python-binding-only API to retrieve a read-only output buffer and its tensor info
 *
 * After nnfw_run has been called, the session has already resized or allocated the internal output
 * buffer to match the latest output dimensions. This API simply retrieves that internal buffer
 * pointer and the corresponding tensor info without performing any allocation itself.
 *
 * Note: this function is intended for Python binding only. The buffer is managed internally by the
 * session and must not be modified by the caller. In Python, wrap the pointer as a NumPy array and
 * set array.flags.writeable = False to enforce read-only access.
 *
 * Important: To use this API, you must call
 * nnfw_set_prepare_config(session, NNFW_ENABLE_INTERNAL_OUTPUT_ALLOC, "true")
 * before calling nnfw_prepare().
 *
 * @param[in]    session     The session object
 * @param[in]    index       Output tensor index
 * @param[out]   out_info    nnfw_tensorinfo to be filled with the latest shape/type information
 * @param[out]   out_buffer  Pointer to a const buffer managed by the session
 * @return       NNFW_STATUS_NO_ERROR on success, otherwise an error code
 */
NNFW_STATUS nnfw_get_output(nnfw_session *session, uint32_t index, nnfw_tensorinfo *out_info,
                            const void **out_buffer);

#endif // __NNFW_INTERNAL_H__
