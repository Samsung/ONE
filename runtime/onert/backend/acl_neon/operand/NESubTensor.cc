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

#include "NESubTensor.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{
namespace operand
{

NESubTensor::NESubTensor(INETensor *parent, const arm_compute::TensorShape &tensor_shape,
                         const arm_compute::Coordinates &coords, size_t rank, bool extend_parent)
  : INETensor{rank}, _ne_sub_tensor(std::make_shared<arm_compute::SubTensor>(
                       parent->handle(), tensor_shape, coords, extend_parent))
{
  // DO NOTHING
}

const arm_compute::SubTensor *NESubTensor::handle() const { return _ne_sub_tensor.get(); }

arm_compute::SubTensor *NESubTensor::handle() { return _ne_sub_tensor.get(); }

} // namespace operand
} // namespace acl_neon
} // namespace backend
} // namespace onert
