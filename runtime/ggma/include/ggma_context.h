/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file  ggma_context.h
 * @brief This file defines the GGMA context management API.
 */
#ifndef __GGMA_GGMA_CONTEXT_H__
#define __GGMA_GGMA_CONTEXT_H__

#include "ggma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a GGMA inference context.
 *
 * A GGMA context represents the environment for inference.
 * It includes model, computation graph, KV caches, and other resources for token generation.
 */
typedef struct ggma_context ggma_context;

/**
 * @brief Creates a context from a package path.
 *
 * This function initializes a context for inference from the specified package path.
 * Once the context is no longer needed, it must be destroyed by calling
 * {@link ggma_free_context}.
 *
 * @param[out]  context      A pointer to the variable that will receive the new context handle.
 * @param[in]   package_path  The path to the GGMA package directory.
 * @return      @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_create_context(ggma_context **context, const char *package_path);

/**
 * @brief Closes a GGMA context and releases its resources.
 *
 * This function deallocates all resources associated with the context, including
 * the GGMA package it was created with. After this function returns, the context
 * handle is invalid and must not be used.
 *
 * @param[in] context  The context to close.
 * @return    @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_free_context(ggma_context *context);

#ifdef __cplusplus
}
#endif

#endif // __GGMA_GGMA_CONTEXT_H__
