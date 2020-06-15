/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <string>
#include <vector>

#ifndef __NNFW_BENCHMARK_TYPES_H__
#define __NNFW_BENCHMARK_TYPES_H__

namespace benchmark
{

enum MemoryType
{
  RSS,
  HWM,
  PSS,
  END_OF_MEM_TYPE
};

inline std::string getMemoryTypeString(MemoryType type)
{
  switch (type)
  {
    case MemoryType::RSS:
      return "RSS";
    case MemoryType::HWM:
      return "HWM";
    case MemoryType::PSS:
      return "PSS";
    default:
      return "END_OF_MEM_TYPE";
  }
}

inline std::string getMemoryTypeString(int type)
{
  return getMemoryTypeString(static_cast<MemoryType>(type));
}

enum PhaseEnum
{
  MODEL_LOAD,
  PREPARE,
  WARMUP,
  EXECUTE,
  END_OF_PHASE,
};

inline std::string getPhaseString(PhaseEnum phase)
{
  switch (phase)
  {
    case PhaseEnum::MODEL_LOAD:
      return "MODEL_LOAD";
    case PhaseEnum::PREPARE:
      return "PREPARE";
    case PhaseEnum::WARMUP:
      return "WARMUP";
    case PhaseEnum::EXECUTE:
      return "EXECUTE";
    default:
      return "END_OF_PHASE";
  }
}

inline std::string getPhaseString(int phase)
{
  return getPhaseString(static_cast<PhaseEnum>(phase));
}

inline PhaseEnum getPhaseEnum(const std::string &phase)
{
  if (phase.compare("MODEL_LOAD") == 0)
    return PhaseEnum::MODEL_LOAD;
  if (phase.compare("PREPARE") == 0)
    return PhaseEnum::PREPARE;
  if (phase.compare("WARMUP") == 0)
    return PhaseEnum::WARMUP;
  if (phase.compare("EXECUTE") == 0)
    return PhaseEnum::EXECUTE;
  return PhaseEnum::END_OF_PHASE;
}

const std::vector<std::string> gPhaseStrings{"MODEL_LOAD", "PREPARE", "WARMUP", "EXECUTE"};

enum FigureType
{
  MEAN,
  MAX,
  MIN,
  GEOMEAN,
  END_OF_FIG_TYPE
};

inline std::string getFigureTypeString(FigureType type)
{
  switch (type)
  {
    case MEAN:
      return "MEAN";
    case MAX:
      return "MAX";
    case MIN:
      return "MIN";
    case GEOMEAN:
      return "GEOMEAN";
    default:
      return "END_OF_FIG_TYPE";
  }
}

inline std::string getFigureTypeString(int type)
{
  return getFigureTypeString(static_cast<FigureType>(type));
}

} // namespace benchmark

#endif // __NNFW_BENCHMARK_TYPES_H__
