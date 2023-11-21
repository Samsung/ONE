/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_LOSS_H__
#define __ONERT_IR_OPERATION_LOSS_H__

#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

class Loss : public Operation
{
public:
  enum Input
  {
    Y_PRED = 0,
    Y_TRUE = 1
    // TODO Add more inputs if necessary
  };

  // NOTE It is not yet determined how to get the information of the previous activation when
  //      generating kernels of Loss operation for each backend. If it is determined to get it
  //      from the object of this class, we have to consider whether to change this enum class.
  enum class Type
  {
    MEAN_SQUARED_ERROR,
    CATEGORICAL_CROSSENTROPY
  };

  enum class ReductionType
  {
    SUM_OVER_BATCH_SIZE,
    SUM,
    NONE, // TODO Remove this
  };

  struct CategoricalCrossentropyParam
  {
    int32_t axis;
    float label_smoothing;
  };

  struct SparseCrossentropyParam
  {
    int32_t igore_class;
  };

  struct Param
  {
    Type op_type;
    ReductionType reduction_type;
    union TypeParam {
      CategoricalCrossentropyParam cce;
      SparseCrossentropyParam sce;
    } type_param;
    // TODO Add more params if necessary
    Param() : op_type(Type::MEAN_SQUARED_ERROR), reduction_type(ReductionType::SUM_OVER_BATCH_SIZE)
    {
    }
  };

public:
  Loss(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs, const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  std::string name() const override;
  OpCode opcode() const final { return OpCode::Loss; }

public:
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_LOSS_H__
