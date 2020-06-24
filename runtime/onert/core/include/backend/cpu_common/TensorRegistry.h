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

#ifndef __ONERT_BACKEND_CPU_COMMON_TENSOR_REGISTRY__
#define __ONERT_BACKEND_CPU_COMMON_TENSOR_REGISTRY__

#include "backend/ITensorRegistry.h"
#include "Tensor.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

#if 0
class TensorRegistry : public ITensorRegistry, public ir::OperandIndexMap<std::shared_ptr<Tensor>>
{
public:
  /**
   * @brief Returns pointer of ITensor
   * @note  Returned tensor cannot be used longer than dynamic tensor manager
   */
  std::shared_ptr<ITensor> getITensor(const ir::OperandIndex &ind) override { return at(ind); }
};
#endif

using TensorRegistry = PortableTensorRegistryTemplate<cpu_common::Tensor>;

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_TENSOR_REGISTRY__
