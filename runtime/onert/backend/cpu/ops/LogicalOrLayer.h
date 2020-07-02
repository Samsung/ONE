/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in riting, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_BACKEND_CPU_OPS_LOGICAL_OR_LAYER_H__
#define __ONERT_BACKEND_CPU_OPS_LOGICAL_OR_LAYER_H__

#include <backend/IPortableTensor.h>

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
class LogicalOrLayer : public ::onert::exec::IFunction
{
public:
  LogicalOrLayer() : _lhs(nullptr), _rhs(nullptr), _output(nullptr)
  {
    // Nothing
  }

public:
  void configure(const IPortableTensor *_lhs, const IPortableTensor *_rhs, IPortableTensor *output);

  void run();

private:
  void lorBool8();

private:
  const IPortableTensor *_lhs;
  const IPortableTensor *_rhs;
  IPortableTensor *_output;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_LOGICAL_OR_LAYER_H__
