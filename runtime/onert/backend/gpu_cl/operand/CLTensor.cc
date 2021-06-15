/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CLTensor.h"

#include "open_cl/Buffer.h"
#include "open_cl/ClContext.h"
#include "open_cl/Tensor.h"
#include "open_cl/TensorType.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
namespace operand
{

CLTensor::CLTensor(size_t rank, ir::Shape shape, CLCommandQueue *queue, size_t num_uses)
  : ICLTensor{rank, shape, queue}, _tensor(std::make_shared<Tensor>()), _num_uses{num_uses}
{
}

const Tensor *CLTensor::handle() const { return _tensor.get(); }

Tensor *CLTensor::handle() { return _tensor.get(); }

void CLTensor::setBuffer(void *host_ptr) { (void)host_ptr; }

} // namespace operand
} // namespace gpu_cl
} // namespace backend
} // namespace onert
