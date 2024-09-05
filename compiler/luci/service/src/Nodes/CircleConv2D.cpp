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

#include "HelperConv2Ds.h"
#include "luci/Service/CircleShapeInference.h"

#include "CircleCloneNode.h"
#include "CircleShapeInferenceHelper.h"

#include <luci/Log.h>

namespace luci
{

luci::CircleNode *CloneNodeLet<CN::ABC>::visit(const luci::CircleConv2D *node)
{
  if (node->fusedActivationFunction() == luci::FusedActFunc::UNDEFINED)
    return nullptr;
  if (node->padding() == luci::Padding::UNDEFINED)
    return nullptr;

  auto *cloned = _graph->nodes()->create<luci::CircleConv2D>();
  if (cloned != nullptr)
  {
    cloned->fusedActivationFunction(node->fusedActivationFunction());
    cloned->padding(node->padding());
    cloned->stride()->h(node->stride()->h());
    cloned->stride()->w(node->stride()->w());
    cloned->dilation()->h(node->dilation()->h());
    cloned->dilation()->w(node->dilation()->w());
  }
  return cloned;
}

namespace sinf
{

loco::TensorShape Algorithm::visit(const luci::CircleConv2D *node)
{
  LOGGER(l);

  std::cout << "CircleConv2D ShapeInf" << std::endl;
  
  auto ifm_shape = luci::shape_get(node->input()).as<loco::TensorShape>();  // in NHWC
  auto ker_shape = luci::shape_get(node->filter()).as<loco::TensorShape>(); // in OHWI

  assert(ifm_shape.rank() == 4);
  assert(ker_shape.rank() == 4);
  assert(ifm_shape.dim(3) == ker_shape.dim(3));

  loco::TensorShape ofm_shape = conv2d_output_shape(node);
  // Height and width have already been determined by conv2d_output_shape
  ofm_shape.dim(0) = ifm_shape.dim(0);
  ofm_shape.dim(3) = ker_shape.dim(0);

  INFO(l) << "[luci] CircleConv2D ShapeInf ifm(" << ifm_shape.rank() << ") ker(" << ker_shape.rank()
          << ") output(" << ofm_shape.dim(0).value() << "," << ofm_shape.dim(1).value() << ","
          << ofm_shape.dim(2).value() << "," << ofm_shape.dim(3).value() << ") " << node->name()
          << std::endl;

  std::cout << "CircleConv2D ShapeInf end" << std::endl;

  return ofm_shape;
}

} // namespace sinf

} // namespace luci
