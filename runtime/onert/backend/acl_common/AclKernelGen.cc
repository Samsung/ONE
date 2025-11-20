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

#include "AclKernelGen.h"

namespace onert::backend::acl_common
{

void enableDimCorrection(IACLTensor *tensor)
{
  size_t input_rank = tensor->getShape().rank();
  const_cast<arm_compute::TensorShape &>(tensor->info()->tensor_shape())
    .set(input_rank - 1, tensor->info()->dimension(input_rank - 1), true);
}

void disableDimCorrection(IACLTensor *tensor)
{
  size_t input_rank = tensor->getShape().rank();
  const_cast<arm_compute::TensorShape &>(tensor->info()->tensor_shape())
    .set(input_rank - 1, tensor->info()->dimension(input_rank - 1), false);
}

} // namespace onert::backend::acl_common
