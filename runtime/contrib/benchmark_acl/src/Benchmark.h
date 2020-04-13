/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ACL_BENCHMARK_H__
#define __ACL_BENCHMARK_H__

#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph.h"
#include "arm_compute/core/CL/OpenCL.h"

struct InputAccessor final : public arm_compute::graph::ITensorAccessor
{
    InputAccessor() = default;
    /** Allows instances to move constructed */
    InputAccessor(InputAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(arm_compute::ITensor &tensor) override
    {
      return true;
    }
};

struct OutputAccessor final : public arm_compute::graph::ITensorAccessor
{
    OutputAccessor() = default;
    /** Allows instances to move constructed */
    OutputAccessor(OutputAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(arm_compute::ITensor &tensor) override
    {
      return false;
    }
};

template <typename T> std::unique_ptr<arm_compute::graph::ITensorAccessor> get_accessor()
{
  return std::unique_ptr<T>(new T());
}

class Count
{
public:
  Count();

public:
  uint32_t value(void) const;

private:
  uint32_t _value;
};

inline arm_compute::graph::Target set_target_hint(int target)
{
    if(target == 1 && arm_compute::opencl_is_available())
    {
        // If type of target is OpenCL, check if OpenCL is available and initialize the scheduler
        return arm_compute::graph::Target::CL;
    }
    else
    {
        return arm_compute::graph::Target::NEON;
    }
}

void run_benchmark(arm_compute::graph::frontend::Stream &graph);

#endif
