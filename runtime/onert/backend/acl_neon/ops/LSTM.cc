/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "../KernelGenerator.h"
#include "../Validator.h"

#include <AclKernelGen.h>

namespace onert::backend::acl_neon
{

void Validator::visit(const ir::operation::LSTM &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::LSTM &node)
{
  _return_fn = acl_common::kernelGenLSTM<acl_common::AclFunction, ::arm_compute::ITensor,
                                         ::arm_compute::NELSTMLayer>(node, _ctx, _tensor_reg);
}

} // namespace onert::backend::acl_neon
