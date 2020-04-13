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

#ifndef __MORPH_NNAPI_H__
#define __MORPH_NNAPI_H__

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/feature/Shape.h>
#include <nncc/core/ADT/kernel/Shape.h>

namespace morph
{
namespace nnapi
{

nncc::core::ADT::tensor::Shape as_tensor_shape(const nncc::core::ADT::feature::Shape &);
nncc::core::ADT::tensor::Shape as_tensor_shape(const nncc::core::ADT::kernel::Shape &);

nncc::core::ADT::feature::Shape as_feature_shape(const nncc::core::ADT::tensor::Shape &);
nncc::core::ADT::kernel::Shape as_kernel_shape(const nncc::core::ADT::tensor::Shape &);

} // namespace nnapi
} // namespace morph

#endif // __MORPH_NNAPI_H__
