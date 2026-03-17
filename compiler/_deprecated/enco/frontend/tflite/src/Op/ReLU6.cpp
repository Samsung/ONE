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

#include "ReLU6.h"

#include "IRBuilder.h"
#include "GraphBuilder.h"

#include <morph/tflite.h>
#include <coco/IR/Module.h>
#include <coco/IR/FeatureLayouts.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::tflite;

namespace tflimport
{

void ReLU6GraphBuilder::build(const tflite::Operator *op, GraphBuilderContext *context) const
{
  assert(context != nullptr); // check if init(..) is called

  coco::Module *m = context->m();
  coco::Block *blk = context->block();
  TensorContext &tensor_context = context->tensor();
  TensorBags &bags = context->bags();

  IndexVector opinputs = as_index_vector(op->inputs());
  IndexVector opoutputs = as_index_vector(op->outputs());

  // these are fixed in tflite
  // input index 0 : input feature
  // output index 0 : output feature
  assert(opinputs.size() == 1);
  assert(opoutputs.size() == 1);

  int ifm_idx = opinputs.at(0);
  int ofm_idx = opoutputs.at(0);

  const tensor::Shape &ifm_shape = tensor_context.shape(ifm_idx);
  const tensor::Shape &ofm_shape = tensor_context.shape(ofm_idx);

  // Create an object for an input feature map
  coco::FeatureObject *ifm_obj = m->entity()->object()->create<coco::FeatureObject>();
  coco::Bag *ifm_bag = bags.bag(ifm_idx);
  ifm_obj->bag(ifm_bag);
  ifm_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ifm_shape)));

  // Create an object for an output feature map
  coco::FeatureObject *ofm_obj = m->entity()->object()->create<coco::FeatureObject>();
  coco::Bag *ofm_bag = bags.bag(ofm_idx);
  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ofm_shape)));

  // Create a Load op
  auto coco_load = op_builder(m).load(ifm_obj).pop();

  // Create a ReLU6
  auto coco_relu6 = m->entity()->op()->create<coco::ReLU6>();

  // Link ops
  coco_relu6->arg(coco_load);

  // Create an Eval instruction
  auto eval_ins = instr_builder(m).eval(ofm_obj, coco_relu6);

  // Append the instruction to the block
  blk->instr()->append(eval_ins);
}

} // namespace tflimport
