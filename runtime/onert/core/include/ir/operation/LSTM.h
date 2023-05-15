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
#ifndef __ONERT_IR_OPERATION_LSTM_H__
#define __ONERT_IR_OPERATION_LSTM_H__

#include "ir/InternalType.h"
#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

// This operation supports only unidirectional sequence lstm
class LSTM : public Operation
{
public:
  enum Input
  {
    INPUT = 0,
    INPUT_TO_INPUT_WEIGHTS = 1,
    INPUT_TO_FORGET_WEIGHTS = 2,
    INPUT_TO_CELL_WEIGHTS = 3,
    INPUT_TO_OUTPUT_WEIGHTS = 4,
    RECURRENT_TO_INPUT_WEIGHTS = 5,
    RECURRENT_TO_FORGET_WEIGHTS = 6,
    RECURRENT_TO_CELL_WEIGHTS = 7,
    RECURRENT_TO_OUTPUT_WEIGHTS = 8,
    CELL_TO_INPUT_WEIGHTS = 9,
    CELL_TO_FORGET_WEIGHTS = 10,
    CELL_TO_OUTPUT_WEIGHTS = 11,
    INPUT_GATE_BIAS = 12,
    FORGET_GATE_BIAS = 13,
    CELL_BIAS = 14,
    OUTPUT_GATE_BIAS = 15,
    PROJECTION_WEIGHTS = 16,
    PROJECTION_BIAS = 17,
    OUTPUT_STATE_IN = 18,
    CELL_STATE_IN = 19,
    INPUT_LAYER_NORMALIZATION_WEIGHTS = 20,
    FORGET_LAYER_NORMALIZATION_WEIGHTS = 21,
    CELL_LAYER_NORMALIZATION_WEIGHTS = 22,
    OUTPUT_LAYER_NORMALIZATION_WEIGHTS = 23,
  };

  enum Output
  {
    SCRATCH_BUFFER = 0,
    OUTPUT_STATE_OUT = 1,
    CELL_STATE_OUT = 2,
    OUTPUT = 3
  };

  struct Param
  {
    Activation activation;
    float cell_threshold;
    float projection_threshold;
    bool time_major;
  };

public:
  LSTM(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs, const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  void accept(MutableOperationVisitor &v) override;
  std::string name() const override;
  OpCode opcode() const final { return OpCode::LSTM; }

public:
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_LSTM_H__
