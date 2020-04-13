/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Div.h"

#include "Convert.h"
#include "IRBuilder.h"
#include "GraphBuilder.h"
#include "Padding.h"
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

void DivGraphBuilder::build(const tflite::Operator *op, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  coco::Module *m = context->m();
  coco::Block *blk = context->block();
  TensorContext &tensor_context = context->tensor();
  TensorBags &bags = context->bags();

  IndexVector opinputs = as_index_vector(op->inputs());
  IndexVector opoutputs = as_index_vector(op->outputs());

  // these are fixed in tflite
  // input index 0 : numerator
  // input index 1 : denominator
  // output index 0 : result
  assert(opinputs.size() == 2);
  assert(opoutputs.size() == 1);

  tflite::ActivationFunctionType activation;
  if (auto *options = op->builtin_options_as_DivOptions())
  {
    activation = options->fused_activation_function();
  }
  else
  {
    activation = tflite::ActivationFunctionType_NONE;
  }

  // TODO activation, e.g. ReLU
  assert(activation == tflite::ActivationFunctionType_NONE);

  auto num_idx = opinputs.at(0);
  auto denom_idx = opinputs.at(1);
  auto out_idx = opoutputs.at(0);

  const tensor::Shape &num_shape = tensor_context.shape(num_idx);
  const tensor::Shape &denom_shape = tensor_context.shape(denom_idx);
  const tensor::Shape &out_shape = tensor_context.shape(out_idx);

  // TODO Now input/output assumes Feature map, but Div should support generic object type
  // Create an object for an input
  auto *num_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *num_bag = bags.bag(num_idx);
  num_obj->bag(num_bag);
  num_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(num_shape)));

  auto *denom_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *denom_bag = bags.bag(denom_idx);
  denom_obj->bag(denom_bag);
  denom_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(denom_shape)));

  // Create an object for an output
  auto *out_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *out_bag = bags.bag(out_idx);
  out_obj->bag(out_bag);
  out_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(out_shape)));

  // Create a Load ops for each input
  auto coco_load_num = op_builder(m).load(num_obj).pop();
  auto coco_load_denom = op_builder(m).load(denom_obj).pop();

  // Create a Div op
  auto coco_div = m->entity()->op()->create<coco::Div>();

  // Link ops
  coco_div->left(coco_load_num);
  coco_div->right(coco_load_denom);

  // Create an Eval instruction
  auto eval_ins = instr_builder(m).eval(out_obj, coco_div);

  // Append the instruction to the block
  blk->instr()->append(eval_ins);
}

} // namespace tflimport
