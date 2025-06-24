/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_RUNTVN_H__
#define __ONERT_IR_OPERATION_RUNTVN_H__

#include "ir/Operation.h"

namespace onert::ir::operation
{

/**
 * @brief Class to represent running TVN model operation
 * @note  This operation is virtual operation which is not used on real circle model,
 *        but used internally to represent running TVN model which is binary model format for
 *        specific backend - trix backend.
 *        RunTVN operation node is created when loading TVN model from given path by TVNLoader.
 */
class RunTVN : public Operation
{
public:
  struct Param
  {
    std::string binary_path;
    std::vector<ir::Shape> origin_input_shapes;
    std::vector<ir::Shape> origin_output_shapes;
  };

public:
  RunTVN(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
         const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::RunTVN; }
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace onert::ir::operation

#endif // __ONERT_IR_OPERATION_RUNTVN_H__
