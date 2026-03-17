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

#include "Conv2D.h"

#include "Convert.h"
#include "IRBuilder.h"
#include "GraphBuilder.h"
#include "Padding.h"
#include "Activation.h"

#include <morph/tflite.h>
#include <coco/IR/Module.h>
#include <coco/IR/KernelLayouts.h>
#include <coco/IR/FeatureLayouts.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::tflite;

namespace tflimport
{

bool Conv2DGraphBuilder::validate(const tflite::Operator *op) const
{
  auto const options = op->builtin_options_as_Conv2DOptions();

  if ((options->stride_h() == 0) || (options->stride_w() == 0))
  {
    return false;
  }

  return true;
}

void Conv2DGraphBuilder::build(const tflite::Operator *op, GraphBuilderContext *context) const
{
  assert(context != nullptr);

  // preparation
  coco::Module *m = context->m();
  coco::Data *d = context->d();
  coco::Block *blk = context->block();
  TensorContext &tensor_context = context->tensor();
  TensorBags &bags = context->bags();
  TflBufferContext &buffer_context = context->buffer();
  const tflite::SubGraph *graph = context->graph();
  IndexVector opinputs = as_index_vector(op->inputs());
  IndexVector opoutputs = as_index_vector(op->outputs());

  // these are fixed in tflite
  // input index 0 : input feature
  // input index 1 : kernel
  // input index 2 : bias (optional)
  bool hasBias = (opinputs.size() == 3);
  assert(opinputs.size() == 2 || hasBias);
  assert(opoutputs.size() == 1);

  int ifm_idx = opinputs.at(0);
  int ker_idx = opinputs.at(1);
  int ofm_idx = opoutputs.at(0);

  const tensor::Shape &ifm_shape = tensor_context.shape(ifm_idx);
  const tensor::Shape &ofm_shape = tensor_context.shape(ofm_idx);
  const tensor::Shape &ker_shape = tensor_context.shape(ker_idx);

  // Create an input feature map object
  auto *ifm_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *ifm_bag = bags.bag(ifm_idx);
  ifm_obj->bag(ifm_bag);
  ifm_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ifm_shape)));

  // Create an an output feature map object
  auto *ofm_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *ofm_bag = bags.bag(ofm_idx);
  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ofm_shape)));

  // Create an kernel object
  auto *ker_obj = m->entity()->object()->create<coco::KernelObject>();
  auto *ker_bag = bags.bag(ker_idx);
  ker_obj->bag(ker_bag);
  ker_obj->layout(coco::KernelLayouts::NHWC::create(as_kernel_shape(ker_shape)));

  // Create a Load op
  auto load = op_builder(m).load(ifm_obj).pop();

  // Create a Conv2D op
  auto coco_conv2d = m->entity()->op()->create<coco::Conv2D>();

  // populating Conv2D objects and options such as stride and padding
  coco_conv2d->ker(ker_obj);

  auto *conv_params = op->builtin_options_as_Conv2DOptions();

  coco_conv2d->stride()->vertical(conv_params->stride_h());
  coco_conv2d->stride()->horizontal(conv_params->stride_w());

  // conv_params->padding() to left, top, right, bottom
  coco::Padding2D padding = conv2D_padding(conv_params, ifm_shape, ker_shape);

  coco_conv2d->pad()->top(padding.top());
  coco_conv2d->pad()->bottom(padding.bottom());
  coco_conv2d->pad()->left(padding.left());
  coco_conv2d->pad()->right(padding.right());

  // Link ops
  coco_conv2d->arg(load);

  // Object to store Conv2D output
  auto *conv2d_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *conv2d_bag = m->entity()->bag()->create(num_elements(ofm_shape));
  conv2d_obj->bag(conv2d_bag);
  conv2d_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ofm_shape)));

  // Create an Eval instruction for Conv2D
  auto conv2d_ins = instr_builder(m).eval(conv2d_obj, coco_conv2d);

  // Append the instruction to the block
  blk->instr()->append(conv2d_ins);

  // Last Object to make a copy to Output Object
  coco::FeatureObject *last_obj = conv2d_obj;

  if (hasBias)
  {
    // When there is a bias, use btmp_obj as bias add output
    // Bias is adding last_obj with bias weight values
    auto *btmp_obj = m->entity()->object()->create<coco::FeatureObject>();
    auto *btmp_bag = m->entity()->bag()->create(num_elements(ofm_shape));
    btmp_obj->bag(btmp_bag);
    btmp_obj->layout(coco::FeatureLayouts::BHWC::create(ofm_obj->shape()));

    int bias_idx = opinputs.at(2);

    // Create an object for bias
    auto bias_obj = m->entity()->object()->create<coco::FeatureObject>();
    coco::Bag *bias_bag = bags.bag(bias_idx);
    bias_obj->bag(bias_bag);
    bias_obj->layout(coco::FeatureLayouts::BC::create(ofm_obj->shape()));

    // Create Op of conv2d output (last_obj) + bias values(bias_obj)
    auto bias_add = op_builder(m).load(last_obj).load(bias_obj).add().pop();

    // Create Instr as bias add result write to btmp_obj
    auto bias_add_ins = instr_builder(m).eval(btmp_obj, bias_add);

    // Append the instruction
    blk->instr()->append(bias_add_ins);

    // Update last_obj to btmp_obj
    last_obj = btmp_obj;
  }

  // fused activation
  coco::FeatureObject *act_output =
    build_activation(conv_params->fused_activation_function(), blk, last_obj);

  // Create Copy Instr of last_obj to Output Object
  auto copy_ins = instr_builder(m).copy(ofm_obj, act_output);
  blk->instr()->append(copy_ins);
}

} // namespace tflimport
