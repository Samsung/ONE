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

#ifndef __ONERT_BACKEND_ACL_COMMON_ACL_ACTIVATION_BUILDER_H__
#define __ONERT_BACKEND_ACL_COMMON_ACL_ACTIVATION_BUILDER_H__

#include <memory>

#include <ir/InternalType.h>
#include <exec/IFunction.h>
#include <exec/NopFunction.h>

#include "Convert.h"

namespace onert
{
namespace backend
{
namespace acl_common
{

template <typename T_Tensor, typename T_ActivationLayer, typename T_ExecFunction>
class AclActivationBuilder
{
private:
  static std::unique_ptr<exec::IFunction> generateReLU(T_Tensor *ifm_alloc);
  static std::unique_ptr<exec::IFunction> generateReLU1(T_Tensor *ifm_alloc);
  static std::unique_ptr<exec::IFunction> generateReLU6(T_Tensor *ifm_alloc);

public:
  static std::unique_ptr<exec::IFunction> generate(ir::Activation code, T_Tensor *ifm_alloc);
};

template <typename T_Tensor, typename T_ActivationLayer, typename T_ExecFunction>
std::unique_ptr<exec::IFunction>
AclActivationBuilder<T_Tensor, T_ActivationLayer, T_ExecFunction>::generateReLU(T_Tensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
    ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};

  auto fn = std::make_unique<T_ActivationLayer>();

  fn->configure(ifm_alloc, nullptr, act_info);

  return asFunction<T_ExecFunction>(std::move(fn));
}

template <typename T_Tensor, typename T_ActivationLayer, typename T_ExecFunction>
std::unique_ptr<exec::IFunction>
AclActivationBuilder<T_Tensor, T_ActivationLayer, T_ExecFunction>::generateReLU1(
  T_Tensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
    ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};

  auto fn = std::make_unique<T_ActivationLayer>();

  fn->configure(ifm_alloc, nullptr, act_info);

  return asFunction<T_ExecFunction>(std::move(fn));
}

template <typename T_Tensor, typename T_ActivationLayer, typename T_ExecFunction>
std::unique_ptr<exec::IFunction>
AclActivationBuilder<T_Tensor, T_ActivationLayer, T_ExecFunction>::generateReLU6(
  T_Tensor *ifm_alloc)
{
  const ::arm_compute::ActivationLayerInfo act_info{
    ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.0f};

  auto fn = std::make_unique<T_ActivationLayer>();

  fn->configure(ifm_alloc, nullptr, act_info);

  return asFunction<T_ExecFunction>(std::move(fn));
}

template <typename T_Tensor, typename T_ActivationLayer, typename T_ExecFunction>
std::unique_ptr<exec::IFunction>
AclActivationBuilder<T_Tensor, T_ActivationLayer, T_ExecFunction>::generate(ir::Activation code,
                                                                            T_Tensor *ifm_alloc)
{
  switch (code)
  {
    case ir::Activation::NONE:
    {
      return std::make_unique<exec::NopFunction>();
    }
    case ir::Activation::RELU:
    {
      return generateReLU(ifm_alloc);
    }
    case ir::Activation::RELU1:
    {
      return generateReLU1(ifm_alloc);
    }
    case ir::Activation::RELU6:
    {
      return generateReLU6(ifm_alloc);
    }
    default:
    {
      throw std::runtime_error("Not supported, yet");
    }
  }
}

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_ACL_ACTIVATION_BUILDER_H__
