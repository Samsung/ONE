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

#ifndef __NEURUN_BACKEND_CPU_KERNEL_CONCATLAYER_H__
#define __NEURUN_BACKEND_CPU_KERNEL_CONCATLAYER_H__

#include "../operand/Tensor.h"

#include <exec/IFunction.h>

namespace neurun
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class ConcatLayer : public ::neurun::exec::IFunction
{
public:
  ConcatLayer();

public:
  void concatenationFloat32();

  void concatenationQuant8();

  void configure(const std::vector<const operand::Tensor *> &inputs, int32_t axis,
                 operand::Tensor *output);

  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  std::vector<const operand::Tensor *> _inputs;
  operand::Tensor *_output;
  int32_t _axis;
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_CPU_KERNEL_CONCATLAYER_H__
