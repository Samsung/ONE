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

#ifndef __ONERT_BACKEND_CPU_KERNEL_ONEHOTLAYER_H__
#define __ONERT_BACKEND_CPU_KERNEL_ONEHOTLAYER_H__

#include "../operand/Tensor.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class OneHotLayer : public ::onert::exec::IFunction
{
public:
  OneHotLayer()
      : _indices(nullptr), _output(nullptr), _depth(0), _on_value(1), _off_value(0), _axis(-1)
  {
    // DO NOTHING
  }

public:
  void oneHotFloat32();

  void oneHotQuant8();

  void configure(const ITensor *indices, ITensor *output, int32_t depth,
                 float on_value, float off_value, int32_t axis);

  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  const ITensor *_indices;
  ITensor *_output;

  int32_t _depth;
  float _on_value;
  float _off_value;
  int32_t _axis;
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_ONEHOTLAYER_H__
