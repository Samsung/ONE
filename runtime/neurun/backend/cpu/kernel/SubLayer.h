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

#ifndef __NEURUN_BACKEND_CPU_KERNEL_SUBLAYER_H__
#define __NEURUN_BACKEND_CPU_KERNEL_SUBLAYER_H__

#include "../operand/Tensor.h"
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace neurun
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class SubLayer : public ::neurun::exec::IFunction
{
public:
  SubLayer() : _lhs(nullptr), _rhs(nullptr), _output(nullptr)
  {
    // DO NOTHING
  }

public:
  void subFloat32();

  void subQuant8();

  void configure(const operand::Tensor *lhs, const operand::Tensor *rhs,
                 const ir::Activation activation, operand::Tensor *output);

  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  const operand::Tensor *_lhs;
  const operand::Tensor *_rhs;
  operand::Tensor *_output;

  ir::Activation _activation{ir::Activation::NONE};
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace neurun

#endif // __NEURUN_BACKEND_CPU_KERNEL_SUBLAYER_H__
