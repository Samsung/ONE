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

#ifndef __NNFW_BENCHMARK_PHASE_H__
#define __NNFW_BENCHMARK_PHASE_H__

#include <string>
#include <cassert>

namespace benchmark
{

enum class Phase
{
  MODEL_LOAD,
  PREPARE,
  EXECUTE,
};

inline std::string getPhaseString(Phase phase)
{
  switch (phase)
  {
    case Phase::MODEL_LOAD:
      return "MODEL_LOAD";
    case Phase::PREPARE:
      return "PREPARE";
    case Phase::EXECUTE:
      return "EXECUTE";
    default:
      assert(false);
      return "";
  }
}

} // namespace benchmark

namespace std
{

template <> struct hash<benchmark::Phase>
{
  size_t operator()(benchmark::Phase value) const noexcept
  {
    using type = typename std::underlying_type<benchmark::Phase>::type;
    return hash<type>()(static_cast<type>(value));
  }
};

} // namespace std

#endif // __NNFW_BENCHMARK_PHASE_H__
