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

#ifndef __ONERT_IR_OPERATION_RESIZE_NEAREST_NEIGHBOR_H__
#define __ONERT_IR_OPERATION_RESIZE_NEAREST_NEIGHBOR_H__

#include <memory>

#include "ir/Operation.h"

namespace onert
{
namespace ir
{
namespace operation
{

class ResizeNearestNeighbor : public Operation
{
public:
  enum Input
  {
    INPUT = 0,
    SIZE = 1,
  };

  struct Param
  {
    // If the input SIZE exists in inputs, Be height_out and width_out not set. Ignore these params
    int32_t height_out;
    int32_t width_out;
    bool align_corners;
  };

public:
  ResizeNearestNeighbor(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
                        const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  void accept(MutableOperationVisitor &v) override;
  OpCode opcode() const final { return OpCode::ResizeNearestNeighbor; }

public:
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_RESIZE_NEAREST_NEIGHBOR_H__
