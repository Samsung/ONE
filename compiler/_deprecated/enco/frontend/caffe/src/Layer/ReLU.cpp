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

#include "ReLU.h"
#include "IRBuilder.h"

#include <coco/IR/FeatureLayouts.h>

#include <morph/caffe.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::caffe;

namespace caffeimport
{

void ReLUBuilder::build(const ::caffe::LayerParameter &layer, GraphBuilderContext *context) const
{
  coco::Module *module = context->module();
  coco::Block *blk = context->block();
  std::map<std::string, tensor::Shape> &shape_ctx = context->shape_ctx();
  std::map<std::string, coco::Bag *> &bag_ctx = context->bag_ctx();

  assert(layer.bottom().size() == 1);
  assert(layer.top().size() == 1);

  // PReLU is not supported, yet
  // TODO Support PReLU
  assert(!layer.has_relu_param());

  // NOTE The current implementation treats ReLU as Feature op
  // TODO Support ReLU over general tensor
  const auto ifm_name = layer.bottom(0);
  const auto ifm_shape = shape_ctx.at(ifm_name);
  auto ifm_bag = bag_ctx.at(ifm_name);
  auto ifm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ifm_obj->bag(ifm_bag);
  ifm_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ifm_shape)));

  const auto ofm_name = layer.top(0);
  const auto ofm_shape = ifm_shape;
  auto ofm_bag = module->entity()->bag()->create(num_elements(ofm_shape));
  auto ofm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ofm_shape)));

  // Create a Load Op
  auto load = op_builder(module).load(ifm_obj).pop();

  // Create a ReLU op
  auto op = module->entity()->op()->create<coco::ReLU>();

  op->arg(load);

  // Create a Eval instruction
  auto ins = instr_builder(module).eval(ofm_obj, op);

  // Append the instruction to the block
  blk->instr()->append(ins);

  // Update bag and shape context
  bag_ctx[ofm_name] = ofm_bag;
  shape_ctx[ofm_name] = ofm_shape;
}

} // namespace caffeimport
