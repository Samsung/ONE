/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NNC_CORE_BACKEND_INTERPRETER_DECONV2D_IMPL_
#define _NNC_CORE_BACKEND_INTERPRETER_DECONV2D_IMPL_

#include "mir/Attributes.h"
#include "mir/TensorVariant.h"

namespace mir_interpreter
{

/**
 * @brief Transposed convolution (or Deconvolution)
 * @param input The Input tensor
 * @param op The DeConvolution operation object
 *
 * This is basically the backward pass for the convolution operation,
 * hence all the indexing can be deducted by expressing the input index
 * of Conv in terms of it's output index.
 */

void DeConv2D(const mir::TensorVariant &input, const mir::TensorVariant &kernel,
              const mir::Deconv2DOpAttributes &attributes, mir::TensorVariant &output);

} // namespace mir_interpreter

#endif //_NNC_CORE_BACKEND_INTERPRETER_DECONV2D_IMPL_
