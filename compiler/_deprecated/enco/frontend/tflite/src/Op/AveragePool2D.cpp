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

#include "AveragePool2D.h"

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

bool AvgPool2DGraphBuilder::validate(const tflite::Operator *op) const
{
  auto const options = op->builtin_options_as_Pool2DOptions();

  if ((options->stride_h() == 0) || (options->stride_w() == 0))
  {
    return false;
  }

  return true;
}

void AvgPool2DGraphBuilder::build(const tflite::Operator *op, GraphBuilderContext *context) const
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

  // Create a AvgPool2D
  auto coco_avgpool2d = m->entity()->op()->create<coco::AvgPool2D>();
  auto *params = op->builtin_options_as_Pool2DOptions();

  // NOTE For Tensorflow lite, PaddingExcluded is needed
  coco_avgpool2d->divisor(coco::AvgPool2D::Divisor::PaddingExcluded);

  coco_avgpool2d->window()->height(params->filter_height());
  coco_avgpool2d->window()->width(params->filter_width());

  coco_avgpool2d->stride()->vertical(params->stride_h());
  coco_avgpool2d->stride()->horizontal(params->stride_w());

  coco::Padding2D padding =
    pool2D_padding(params, ifm_shape, params->filter_width(), params->filter_height());

  coco_avgpool2d->pad()->top(padding.top());
  coco_avgpool2d->pad()->bottom(padding.bottom());
  coco_avgpool2d->pad()->left(padding.left());
  coco_avgpool2d->pad()->right(padding.right());

  // Link ops
  coco_avgpool2d->arg(coco_load);

  // Create an Eval instruction
  auto ins = instr_builder(m).eval(ofm_obj, coco_avgpool2d);

  // Append the instruction to the block
  blk->instr()->append(ins);

  // TODO activation, e.g., relu
  assert(params->fused_activation_function() ==
         tflite::ActivationFunctionType::ActivationFunctionType_NONE);
}

} // namespace tflimport
