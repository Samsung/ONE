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

#ifndef __ONERT_BACKEND_ACL_COMMON_ACLBACKEND_CONTEXT_H__
#define __ONERT_BACKEND_ACL_COMMON_ACLBACKEND_CONTEXT_H__

#include <backend/BackendContext.h>
#include <ir/Index.h>
#include <ir/OperandIndexMap.h>
#include <ir/OperandIndexSequence.h>
#include <util/logging.h>

#include <cl_common/BackendContext.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

// TODO Find better way to handle common code (reduce template)
template <typename T_TensorBuilder, typename T_ConstantInitializer, typename T_KernelGenerator,
          typename T_Optimizer>
class AclBackendContext
  : public onert::backend::cl_common::BackendContext<T_TensorBuilder, T_ConstantInitializer,
                                                     T_KernelGenerator>
{
public:
  AclBackendContext(const Backend *backend, ContextData &&data,
                    std::shared_ptr<ITensorRegistry> tensor_registry = nullptr,
                    std::shared_ptr<T_TensorBuilder> tensor_builder = nullptr,
                    std::shared_ptr<T_ConstantInitializer> constant_initializer = nullptr,
                    std::shared_ptr<T_KernelGenerator> kernel_gen = nullptr)
    : onert::backend::cl_common::BackendContext<T_TensorBuilder, T_ConstantInitializer,
                                                T_KernelGenerator>(
        backend, std::move(data), tensor_registry, tensor_builder, constant_initializer, kernel_gen)
  {
    // DO NOTHING
  }

  ITensorRegistry *genTensors() override
  {
    optimizer->optimize();

    this->graph()->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &obj) {
      if (this->external_operands().contains(ind))
        return;

      const auto frontend_layout = obj.info().layout();
      const auto backend_layout = this->operand_layouts().at(ind);
      ir::OperandInfo backend_info{permuteShape(obj.shape(), frontend_layout, backend_layout),
                                   obj.typeInfo(), backend_layout, obj.info().memAllocType(),
                                   obj.isConstant()};
      this->tensor_builder->registerTensorInfo(ind, backend_info);
    });

    // TODO Get compiler options from compiler, and use it rather than getting it from Env
    if (util::getConfigString(util::config::EXECUTOR) == "Linear")
    {
      this->planTensors();
    }
    else
    {
      // For the executors that does not have fixed linear execution order:
      // To make tensors never be deallocated, this is a workaround to use static memory planner
      this->graph()->operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
        if (this->tensor_builder->isRegistered(ind))
          this->tensor_builder->notifyFirstUse(ind);
      });
    }

    this->tensor_builder->prepare();

    return this->tensor_registry.get();
  }

protected:
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info) override
  {
    this->tensor_builder->registerTensorInfo(ind, info);
  }

public:
  // TODO Make it private
  std::shared_ptr<T_Optimizer> optimizer;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_ACLBACKEND_CONTEXT_H__
