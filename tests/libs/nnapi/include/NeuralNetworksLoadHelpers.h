/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

// NOTE To minimize diff with upstream tensorflow, disable clang-format
// clang-format off

// NOTE This header is derived from part of the following file (in TensorFlow v1.12)
//       'externals/tensorflow/tensorflow/contrib/lite/nnapi/NeuralNetworksShim.h'

/**
 * @file NeuralNetworksLoadHelpers.h
 * @brief This file contains functions to load NN API runtime library
 */

#ifndef __NEURAL_NETWORKS_LOAD_HELPER_H__
#define __NEURAL_NETWORKS_LOAD_HELPER_H__

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Print log data
 * @param[in] format    Format string of @c printf
 * @param[in] args      Argument after format string. (Same with @c printf)
 */
#define NNAPI_LOG(format, ...) printf(format "\n", __VA_ARGS__);

/**
 * @brief Create a function pointer named @c fn after loading NN API library
 * @param[in] name    Name of a function
 */
#define LOAD_FUNCTION(name) \
  static name##_fn fn = reinterpret_cast<name##_fn>(nnfw::loadFunction(#name));

/**
 * @brief Run @c fn function. @c fn is created by @ref LOAD_FUNCTION
 * @param[in] args    List of arguments for the function @c fn
 */
#define EXECUTE_FUNCTION(...) \
  if (fn != nullptr) {        \
    fn(__VA_ARGS__);          \
  }

/**
 * @brief Run @c fn function. @c fn is created by @ref LOAD_FUNCTION
 * @param[in] args    List of arguments for the function @c fn
 * @return            the return value of @c fn
 */
#define EXECUTE_FUNCTION_RETURN(...) return fn != nullptr ? fn(__VA_ARGS__) : 0;

namespace nnfw
{

/**
 * @brief Load NN API library
 * @param[in] name path of NN API library
 * @return a symbol table handle of NN API library
 */
inline void* loadLibrary(const char* name) {
  // TODO: change RTLD_LOCAL? Assumes there can be multiple instances of nn
  // api RT
  void* handle = nullptr;
#if 1 //#ifdef __ANDROID__
  handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    NNAPI_LOG("nnapi error: unable to open library %s", name);
  }
#endif
  return handle;
}

/**
 * @brief Load libneuralnetworks.so and return handle of library
 * @return a symbol table handle of NN API library
 */
inline void* getLibraryHandle() {
  static void* handle = loadLibrary("libneuralnetworks.so");
  return handle;
}

/**
 * @brief Return function ptr in libneuralnetworks.so
 * @param[in] name    Name of function
 * @return function pointer
 */
inline void* loadFunction(const char* name) {
  void* fn = nullptr;
  if (getLibraryHandle() != nullptr) {
    fn = dlsym(getLibraryHandle(), name);
  }
  if (fn == nullptr) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
    abort();
  }
  else {
#ifdef _GNU_SOURCE
    Dl_info info;
    if (dladdr(fn, &info))
    {
      NNAPI_LOG("nnapi function '%s' is loaded from '%s' ", name, info.dli_fname);
    }
    else
    {
      NNAPI_LOG("nnapi function '%s' is failed to load", name);
    }
#endif // _GNU_SOURCE
  }
  return fn;
}

/**
 * @brief Check if libneuralnetworks.so can be loaded
 * @return @c true if loading is successful, otherwise @c false.
 */
inline bool NNAPIExists() {
  static bool nnapi_is_available = getLibraryHandle();
  return nnapi_is_available;
}

} // namespace nnfw

#endif // __NEURAL_NETWORKS_LOAD_HELPER_H__
