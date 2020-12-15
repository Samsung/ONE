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

#include "DepthwiseConv2D.h"

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

bool DepthwiseConv2DGraphBuilder::validate(const tflite::Operator *op) const
{
  auto const options = op->builtin_options_as_DepthwiseConv2DOptions();

  if ((options->stride_h() == 0) || (options->stride_w() == 0))
  {
    return false;
  }

  return true;
}

void DepthwiseConv2DGraphBuilder::build(const tflite::Operator *op,
                                        GraphBuilderContext *context) const
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
  tensor::Shape &ker_shape = const_cast<tensor::Shape &>(tensor_context.shape(ker_idx));

  assert(ifm_shape.rank() == 4);
  assert(ofm_shape.rank() == 4);
  assert(ker_shape.rank() == 4);

  assert(ker_shape.dim(0) == 1); // value > 1 was not tested. This value seems 1 in DepthwiseConv2D
  assert(ifm_shape.dim(3) == ofm_shape.dim(3));
  assert(ofm_shape.dim(3) == ker_shape.dim(3));

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

  // Adjust tflite kernel shape [1, h, w, channel_out] for coco::Kernel.
  // coco::Kernel will have kernel.count = channel_out, kernel.depth = 1 ( == ker_shape.dim(0))
  kernel::Shape new_shape{ker_shape.dim(3), 1, ker_shape.dim(1), ker_shape.dim(2)};
  ker_obj->layout(coco::KernelLayouts::NHWC::create(new_shape));

  // Create a kernel overlay for the kernel object
  // TODO : support for other types
  d->f32()->allocate(ker_bag);

  TflBufferContext::TflBuffer<float> buffer = buffer_context.tensor_buffer<float>(graph, ker_idx);

  auto ker_spn = d->f32()->weight(ker_bag);

  // Copy data from tflBuffer of [1, h, w, channel_out] shape to coco::Data, which will be accessed
  // by coco::KernelLayouts::NHWC
  for (auto n = 0; n < new_shape.count(); n++)
  {
    auto tfl_c = n;
    for (auto h = 0; h < new_shape.height(); h++)
    {
      for (auto w = 0; w < new_shape.width(); w++)
      {
        auto hw = new_shape.height() * new_shape.width();
        for (auto c = 0; c < new_shape.depth(); c++)
        {
          auto tfl_n = c;
          auto hwc = hw * new_shape.depth();
          auto wc = new_shape.width() * new_shape.depth();

          ker_spn[n * hwc + h * wc + w * new_shape.depth() + c] =
            buffer.ptr[tfl_n * hw * new_shape.count() + /* new_shape.count() is old c */
                       h * new_shape.width() * new_shape.count() + w * new_shape.count() + tfl_c];
        }
      }
    }
  }

  // Create a Load op
  auto load = op_builder(m).load(ifm_obj).pop();

  // Create a coco::Conv2D op for DepthwiseConv2D
  auto coco_dconv2d = m->entity()->op()->create<coco::Conv2D>();

  // populating objects and options such as stride and padding for DepthwiseConv2D
  coco_dconv2d->ker(ker_obj);

  // setting params passed from TFLITE DepthwiseConv2DOptions
  auto dconv_params = op->builtin_options_as_DepthwiseConv2DOptions();

  assert(dconv_params->depth_multiplier() == 1); // other depth_multiplier was not tested

  coco_dconv2d->group(ifm_obj->asFeature()->shape().depth());

  coco_dconv2d->stride()->vertical(dconv_params->stride_h());
  coco_dconv2d->stride()->horizontal(dconv_params->stride_w());

  coco::Padding2D padding = depthwiseConv2D_padding(dconv_params, ifm_shape, ker_shape);
  coco_dconv2d->pad()->top(padding.top());
  coco_dconv2d->pad()->bottom(padding.bottom());
  coco_dconv2d->pad()->left(padding.left());
  coco_dconv2d->pad()->right(padding.right());

  // Link ops
  coco_dconv2d->arg(load);

  // Object to store output for DepthwiseConv2D
  auto *dconv2d_obj = m->entity()->object()->create<coco::FeatureObject>();
  auto *dconv2d_bag = m->entity()->bag()->create(num_elements(ofm_shape));
  dconv2d_obj->bag(dconv2d_bag);
  dconv2d_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ofm_shape)));

  // Create an Eval instruction for DepthwiseConv2D
  auto dconv2d_ins = instr_builder(m).eval(dconv2d_obj, coco_dconv2d);

  // Append the instruction to the block
  blk->instr()->append(dconv2d_ins);

  // Last Object to make a copy to Output Object
  coco::FeatureObject *last_obj = dconv2d_obj;

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
    build_activation(dconv_params->fused_activation_function(), blk, last_obj);

  // Create Copy Instr of last_obj to Output Object
  auto copy_ins = instr_builder(m).copy(ofm_obj, act_output);
  blk->instr()->append(copy_ins);
}

} // namespace tflimport
