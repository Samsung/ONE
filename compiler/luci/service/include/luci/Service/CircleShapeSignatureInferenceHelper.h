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

ShapeSignature signature_of_input(const luci::CircleNode *node, uint32_t index);

} // namespace ssinf

} // namespace luci

#endif // __LUCI_CIRCLE_SHAPE_SIGNATURE_INFERENCE_HELPER_H__
