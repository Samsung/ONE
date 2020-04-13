/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ANN_IR_OPERAND_H__
#define __ANN_IR_OPERAND_H__

#include "ANN/IR/DType.h"
#include "ANN/IR/Weight.h"

#include <nncc/core/ADT/tensor/Shape.h>

namespace ann
{

class Operand
{
public:
  virtual ~Operand() = default;

public:
  DType dtype(void) const { return _dtype; }
  void dtype(const DType &dtype) { _dtype = dtype; }

  const Weight *weight(void) const { return _weight; }
  void weight(const Weight *weight) { _weight = weight; }

private:
  DType _dtype = DType::UNK;
  const Weight *_weight = nullptr;
};

} // namespace ann

namespace ann
{

/**
 * @brief Plain (non-qunatized) Scalar Operand
 */
struct ScalarOperand final : public Operand
{
};

} // namespace ann

namespace ann
{

/**
 * @brief Plain (non-qunatized) Tensor Operand
 */
struct TensorOperand final : public Operand
{
public:
  TensorOperand(const nncc::core::ADT::tensor::Shape &shape) : _shape{shape}
  {
    // DO NOTHING
  }

public:
  const nncc::core::ADT::tensor::Shape &shape(void) const { return _shape; }

private:
  nncc::core::ADT::tensor::Shape _shape;
};

} // namespace ann

#endif // __ANN_IR_OPERAND_H__
