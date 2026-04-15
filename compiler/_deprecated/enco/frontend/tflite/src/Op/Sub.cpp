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

#include "Sub.h"

#include "Convert.h"
#include "IRBuilder.h"
#include "GraphBuilder.h"
#include "Activation.h"

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

void SubGraphBuilder::build(const tflite::Operator *op, GraphBuilderContext *context) const
{
  assert(context != nullptr); // check if init(..) is called

  coco::Module *m = context->m();
  coco::Block *blk = context->block();
  TensorContext &tensor_context = context->tensor();
  TensorBags &bags = context->bags();

  IndexVector opinputs = as_index_vector(op->inputs());
  IndexVector opoutputs = as_index_vector(op->outputs());

  // these are fixed in tflite
  // input index 0 : left input feature
  // input index 1 : right input feature
  // output index 0 : output feature
  assert(opinputs.size() == 2);
  assert(opoutputs.size() == 1);

  // Default parameter values are referenced from schema_generated.h
  auto *params = op->builtin_options_as_SubOptions();
  tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;

  if (auto *params = op->builtin_options_as_SubOptions())
  {
    activation = params->fused_activation_function();
  }
  assert(activation == tflite::ActivationFunctionType_NONE);

  // Construct a vector of input objects
  std::vector<coco::FeatureObject *> input_objects;

  for (auto &input_index : opinputs)
  {
    // Add objects for input feature map
    const tensor::Shape &input_shape = tensor_context.shape(input_index);
    coco::FeatureObject *input_obj = m->entity()->object()->create<coco::FeatureObject>();
    coco::Bag *input_bag = bags.bag(input_index);
    input_obj->bag(input_bag);
    input_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(input_shape)));

    input_objects.emplace_back(input_obj);
  }

  // Create an object for an output feature map
  int const output_index = opoutputs.at(0);
  const tensor::Shape &output_shape = tensor_context.shape(output_index);
  coco::FeatureObject *output_obj = m->entity()->object()->create<coco::FeatureObject>();
  coco::Bag *output_bag = bags.bag(output_index);
  output_obj->bag(output_bag);
  output_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(output_shape)));

  // Create Load ops
  auto left_load = op_builder(m).load(input_objects[0]).pop();
  auto right_load = op_builder(m).load(input_objects[1]).pop();

  // Create a Sub
  auto coco_sub = m->entity()->op()->create<coco::Sub>();

  coco_sub->left(left_load);
  coco_sub->right(right_load);

  // Create an Eval instruction
  auto eval = instr_builder(m).eval(output_obj, coco_sub);

  // Append the instruction to the block
  blk->instr()->append(eval);

  // TODO activation, e.g., relu
  assert(params->fused_activation_function() ==
         tflite::ActivationFunctionType::ActivationFunctionType_NONE);
}

} // namespace tflimport
