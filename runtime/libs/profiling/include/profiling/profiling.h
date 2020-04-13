/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_MISC_PROFILING_H__
#define __NNFW_MISC_PROFILING_H__

#include <iostream>

namespace tflite
{
namespace profiling
{
class Profiler; // forward declaration
}
}

namespace profiling
{

class Context
{
public:
  Context() : _sync(false), _profiler(nullptr) {}

public:
  const bool &sync(void) const { return _sync; }
  tflite::profiling::Profiler *getProfiler() { return _profiler; }
  void setProfiler(tflite::profiling::Profiler *p) { _profiler = p; }
  void setSync(void) { _sync = true; }

private:
  bool _sync;
  tflite::profiling::Profiler *_profiler;

public:
  static Context &get(void)
  {
    static Context ctx{};
    return ctx;
  }
};

} // namespace profiling
#endif // __NNFW_MISC_PROFILING_H__
