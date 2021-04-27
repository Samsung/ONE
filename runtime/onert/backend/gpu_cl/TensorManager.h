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

#ifndef __ONERT_BACKEND_CL_TENSOR_MANAGER_H__
#define __ONERT_BACKEND_CL_TENSOR_MANAGER_H__

#include "ClMemoryManager.h"
#include "ClTensorManager.h"
#include "../open_cl/ClContext.h"

#include "operand/CLTensor.h"
#include "operand/ICLTensor.h"

#include "util/logging.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

using MemoryManager = ClMemoryManager<operand::ICLTensor, operand::CLTensor>;

using TensorManager = ClTensorManager<operand::ICLTensor, operand::CLTensor>;

inline TensorManager *createTensorManager(CLContext *context)
{
  VERBOSE(createTensorManager) << "ClTensorManager" << std::endl;
  return new TensorManager(new MemoryManager(context), new MemoryManager(context));
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_TENSOR_MANAGER_H__
