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

#include "CLSubTensor.h"

namespace onert
{
namespace backend
{
namespace acl_cl
{
namespace operand
{

CLSubTensor::CLSubTensor(ICLTensor *parent, const arm_compute::TensorShape &tensor_shape,
                         const arm_compute::Coordinates &coords, size_t rank, bool extend_parent)
  : _cl_sub_tensor(std::make_shared<arm_compute::CLSubTensor>(parent->handle(), tensor_shape,
                                                              coords, extend_parent)),
    _rank{rank}
{
  // DO NOTHING
}

const arm_compute::CLSubTensor *CLSubTensor::handle() const { return _cl_sub_tensor.get(); }

arm_compute::CLSubTensor *CLSubTensor::handle() { return _cl_sub_tensor.get(); }

} // namespace operand
} // namespace acl_cl
} // namespace backend
} // namespace onert
