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
 * @file  ggma_package.h
 * @brief This file defines the GGMA package management API.
 */
#ifndef __GGMA_PACKAGE_H__
#define __GGMA_PACKAGE_H__

#include "ggma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a GGMA package.
 *
 * A GGMA package encapsulates all necessary resources for a generative model,
 * including the model itself, default configurations, and vocabulary.
 */
typedef struct ggma_package ggma_package;

/**
 * @brief Creates a GGMA package from a specified directory path.
 *
 * This function loads the necessary components (model, configuration, vocabulary)
 * from the given directory and initializes a GGMA package handle.
 *
 * @param[out] pkg   Pointer to the package object created from the given path
 * @param[in]  path  Path to the directory containing the GGMA package
 * @return     @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_create_package(ggma_package **pkg, const char *path);

/**
 * @brief Frees all resources associated with a GGMA package.
 *
 * After calling {@link ggma_create_context}, the created context
 * assumes ownership of the GGMA package and will automatically free its resources
 * when the context is closed. Therefore, you must not call this function manually
 * on a package that has been used to create a context, as doing so will lead to
 * a double-free error.
 *
 * @param[in] pkg     The GGMA package to free. This handle will be invalid after the call.
 * @return     @c GGMA_STATUS_NO_ERROR if successful, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_free_package(ggma_package *pkg);

#ifdef __cplusplus
}
#endif

#endif // __GGMA_PACKAGE_H__
