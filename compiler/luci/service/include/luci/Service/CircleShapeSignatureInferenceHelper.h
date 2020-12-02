/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_HELPER_H__
#define __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_HELPER_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleShapeSignature.h>

namespace luci
{

namespace ssinf // Namespace for Shape Signature Inference
{

// Return empty signature if all of dimensions are known.
// If one of dimensions is unknown, return signature without change.
ShapeSignature clean_signature(const luci::ShapeSignature &signature);

// Return reduced input_signature with indices and keep_dims.
//  - indices : reduction index
//  - keep_dims : If true, rank is not changed. If false, rank is reduced along indices.
ShapeSignature reduced_signature(const loco::Node *node, const loco::Node *indices, bool keep_dims);

// Return signature of index-th argument of node.
ShapeSignature input_arg_signature(const luci::CircleNode *node, uint32_t index);

} // namespace ssinf

} // namespace luci

#endif // __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_HELPER_H__
