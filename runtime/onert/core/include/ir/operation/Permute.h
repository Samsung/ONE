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

#ifndef __ONERT_IR_OPERATION_PERMUTE_H__
#define __ONERT_IR_OPERATION_PERMUTE_H__

#include "ir/Operation.h"
#include "ir/Layout.h"

namespace onert::backend
{
class BackendContext;
} // namespace onert::backend

namespace onert::ir::operation
{

/**
 * @brief Class to represent Permute operation
 * @note  Permute operation reorders the dimensions of a tensor.
 *
 *        This operation is virtual operation, which is not used on real model, but used internally.
 *        It was introduced to support various model layout (NHWC, NCHW, etc) and backend layout.
 *        But currently, model layout and backend layout are always same as NHWC.
 *        So this operation is used for below cases.
 *        1) Handle model output buffer's special case
 *          1-1) Model output is comes from model constant
 *          1-2) Model output is comes from model input
 *          1-3) Model output shares tensor with other model output(s)
 *        2) Handle shared tensor between different backend
 *        3) Handle when input and/or output layouts are different with model layout
 *        4) Handle when input and/or output data type is different with model data type
 *
 */
class Permute : public Operation
{
public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::Permute; }

public:
  Permute(const OperandIndex &input, const OperandIndex &output, ir::PermuteType type);

public:
  ir::PermuteType getPermuteType() const { return _type; }

private:
  ir::PermuteType _type;
};

} // namespace onert::ir::operation

#endif // __ONERT_IR_OPERATION_PERMUTE_H__
