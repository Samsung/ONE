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

#ifndef __ONERT_IR_OPERATION_SQUEEZE_H__
#define __ONERT_IR_OPERATION_SQUEEZE_H__

#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

class Squeeze : public Operation
{
public:
  enum Input
  {
    INPUT = 0
  };

  struct Param
  {
    // Please see tensorflow/lite/c/builtin_op_data.h and squeeze.cc.
    // tensorflow lite supports only for ndim <= 8.
    int dims[8];
    int ndim;
  };

public:
  Squeeze(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
          const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  void accept(MutableOperationVisitor &v) override;
  OpCode opcode() const final { return OpCode::Squeeze; }
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_SQUEEZE_H__
