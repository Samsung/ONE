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

#ifndef __ONERT_BACKEND_CPU_OPS_BINARYARITHMETICLAYER_H__
#define __ONERT_BACKEND_CPU_OPS_BINARYARITHMETICLAYER_H__

#include <backend/IPortableTensor.h>
#include "OperationUtils.h"

#include <exec/IFunction.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

enum class ArithmeticType
{
  kAdd,
  kSub,
  kMul,
  kDiv,
};

class BinaryArithmeticLayer : public ::onert::exec::IFunction
{
public:
  BinaryArithmeticLayer() : _lhs(nullptr), _rhs(nullptr), _output(nullptr)
  {
    // DO NOTHING
  }

public:
  void configure(const IPortableTensor *lhs, const IPortableTensor *rhs, IPortableTensor *output,
                 const ir::Activation activation, const ArithmeticType arithmetic_type);

  void run() override;

protected:
  const IPortableTensor *_lhs;
  const IPortableTensor *_rhs;
  IPortableTensor *_output;

  std::function<void(const IPortableTensor *, const IPortableTensor *, IPortableTensor *)> _kernel;
};

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPS_BINARYARITHMETICLAYER_H__
