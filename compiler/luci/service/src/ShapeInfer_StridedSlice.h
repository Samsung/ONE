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

#ifndef __SHAPE_INFER_STRIDED_SLICE_H__
#define __SHAPE_INFER_STRIDED_SLICE_H__

#include <luci/IR/CircleNodes.h>

#include <loco/IR/NodeShape.h>

namespace luci
{
namespace sinf
{

loco::TensorShape infer_output_shape(const CircleStridedSlice *node);

} // namespace sinf

} // namespace luci

#endif // __SHAPE_INFER_STRIDED_SLICE_H__
