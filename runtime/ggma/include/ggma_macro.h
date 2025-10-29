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

#include <cstdlib>

/**
 * @file  ggma_macro.h
 * @brief Common macros for GGMA error handling and utilities
 */
#ifndef __GGMA_GGMA_MACRO_H__
#define __GGMA_GGMA_MACRO_H__

#define GGMA_ENSURE(a)               \
  do                                 \
  {                                  \
    if ((a) != GGMA_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

#define GGMA_UNUSED(x) (void)(x)

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif // __GGMA_GGMA_MACRO_H__
