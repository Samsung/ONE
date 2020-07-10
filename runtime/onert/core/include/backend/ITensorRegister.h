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

#ifndef __ONERT_BACKEND_ITENSOR_REGISTER_H__
#define __ONERT_BACKEND_ITENSOR_REGISTER_H__

#include "ir/LowerInfoMap.h"
#include "ITensorBuilder.h"
#include "ir/Layout.h"
#include "ir/OperandIndexSequence.h"
#include "ir/OperandInfo.h"
#include "ir/Operands.h"
#include "ir/OperationVisitor.h"

namespace onert
{
namespace backend
{

class ITensorRegister : public ir::OperationVisitor
{
public:
  virtual ~ITensorRegister() = default;

public:
  void registerTensors(const ir::OpSequence &op_seq, const ir::LowerInfoMap *lower_info_map)
  {
    _current_op_seq_layout = op_seq.getLayout();
    _lower_info_map = lower_info_map;
    assert(_lower_info_map != nullptr);
    assert(tensor_builder().get() != nullptr);
    op_seq.accept(*this);
  }

protected:
  virtual const ir::Operands &operands() const = 0;
  virtual std::shared_ptr<ITensorBuilder> tensor_builder() const = 0;

protected:
#define OP(InternalName)                                                                   \
  void visit(const ir::operation::InternalName &node) override                             \
  {                                                                                        \
    for (const auto &ind : (node.getInputs() | ir::Remove::UNDEFINED) + node.getOutputs()) \
    {                                                                                      \
      defaultRegisterTensorInfo(ind);                                                      \
    }                                                                                      \
  }
#include "ir/Operations.lst"
#undef OP

protected:
  void defaultRegisterTensorInfo(const ir::OperandIndex &index) const
  {
    if (tensor_builder()->isRegistered(index))
    {
      return;
    }

    const auto &obj = operands().at(index);
    const auto frontend_layout = frontendLayout();
    const auto backend_layout = backendLayout(index);
    ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                 obj.typeInfo(), obj.info().memAllocType(), obj.isConstant()};
    tensor_builder()->registerTensorInfo(index, backend_info, backend_layout);
  }

protected:
  ir::Layout frontendLayout() const { return _current_op_seq_layout; }
  ir::Layout backendLayout(const ir::OperandIndex &index) const
  {
    assert(_lower_info_map != nullptr);
    const auto lower_info = _lower_info_map->operand.at(index).get();
    return lower_info->def_factors().getOnlyElement().layout();
  }

private:
  ir::Layout _current_op_seq_layout;
  const ir::LowerInfoMap *_lower_info_map{nullptr};
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ITENSOR_REGISTER_H__
