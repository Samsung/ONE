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

#ifndef __ONERT_BACKEND_CPU_KERNEL_COMPARELAYER_H__
#define __ONERT_BACKEND_CPU_KERNEL_COMPARELAYER_H__

#include "../operand/Tensor.h"

#include <exec/IFunction.h>
#include <ir/operation/Comparison.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace kernel
{

class CompareLayer : public ::onert::exec::IFunction
{
public:
  CompareLayer();

public:
  void compareQuant8();

  void configure(const ITensor *lhs, const ITensor *rhs,
                 const ir::operation::Comparison::ComparisonType op_type, ITensor *output);

  void run();
  void runSync()
  {
    // this abstract method is used just for profiling and called for
    // backend::acl_common::AclFunction
    run();
  }

private:
  const ITensor *_lhs;
  const ITensor *_rhs;
  ITensor *_output;
  ir::operation::Comparison::ComparisonType _op_type;
};

} // namespace kernel
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_KERNEL_COMPARELAYER_H__
