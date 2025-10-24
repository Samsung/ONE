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
 * @file  ggma_types.h
 * @brief This file defines the core types and status codes for GGMA API.
 */
#ifndef __GGMA_TYPES_H__
#define __GGMA_TYPES_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enumeration of status codes returned by GGMA API functions.
 */
typedef enum
{
  /** The operation completed successfully. */
  GGMA_STATUS_NO_ERROR = 0,
  /**
   * A general error occurred.
   * This code is typically used when no more specific error code is applicable.
   */
  GGMA_STATUS_ERROR = 1,
  /** Unexpected null argument is given. */
  GGMA_STATUS_UNEXPECTED_NULL = 2,
  /** The system ran out of memory. */
  GGMA_STATUS_OUT_OF_MEMORY = 4,
  /** The called API function is deprecated and should no longer be used. */
  GGMA_STATUS_DEPRECATED_API = 6,
} GGMA_STATUS;

/* [G] TODO: PyTorch returns token IDs as int64. However, int32_t is likely sufficient. */
typedef int64_t ggma_token;

/* Forward declarations */
typedef struct ggma_package ggma_package;
typedef struct ggma_context ggma_context;

#ifdef __cplusplus
}
#endif

#endif // __GGMA_TYPES_H__
