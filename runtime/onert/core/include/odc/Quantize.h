/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_ODC_QUANTIZE_H__
#define __ONERT_ODC_QUANTIZE_H__

#include <cstdint>
#include <mutex>

namespace onert
{
namespace odc
{
class Quantize
{
public:
  // 1st arg: input file path
  // 2nd arg: output file path
  // 3rd arg: true if q16, false if q8
  // Return value: 0 if success, otherwise error code
  using quantize_t = int (*)(const char *, const char *, bool);
  Quantize(quantize_t quantize_fn) : _quantize(quantize_fn) {}

  int quantize(const char *in, const char *out, bool is_q16)
  {
    // Compile function is thread-unsafe
    std::lock_guard<std::mutex> guard(_lock);
    return static_cast<int32_t>(_quantize(in, out, is_q16));
  }

private:
  quantize_t _quantize;
  std::mutex _lock;
};

} // namespace odc
} // namespace onert

#endif // __ONERT_ODC_QUANTIZE_H__
