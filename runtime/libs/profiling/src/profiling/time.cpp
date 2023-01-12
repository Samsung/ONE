/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// NOTE To minimize diff with upstream tensorflow, disable clang-format
// clang-format off

// NOTE This header is derived from the following file (in TensorFlow v1.12)
//        'externals/tensorflow/tensorflow/lite/profiling/time.cpp
#include "profiling/time.h"

#if defined(_MSC_VER)
#include <chrono>  // NOLINT(build/c++11)
#else
#include <time.h>
#endif

namespace tflite {
namespace profiling {
namespace time {

#if defined(_MSC_VER)

uint64_t NowMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

#else

uint64_t NowMicros() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_nsec) / 1e3 + static_cast<uint64_t>(ts.tv_sec) * 1e6;
}

#endif  // defined(_MSC_VER)

}  // namespace time
}  // namespace profiling
}  // namespace tflite

// clang-format on
