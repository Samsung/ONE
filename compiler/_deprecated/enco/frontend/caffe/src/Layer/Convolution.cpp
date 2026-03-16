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

#include "Convolution.h"
#include "ConvolutionSpec.h"
#include "Convert.h"
#include "IRBuilder.h"

#include <nncc/core/ADT/kernel/Overlay.h>
#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <morph/caffe.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::caffe;

using tensor::num_elements;

namespace caffeimport
{

void ConvolutionBuilder::build(const ::caffe::LayerParameter &layer,
                               GraphBuilderContext *context) const
{
  coco::Module *module = context->module();
  coco::Data *data = context->data();
  coco::Block *blk = context->block();
  std::map<std::string, tensor::Shape> &shape_ctx = context->shape_ctx();
  std::map<std::string, coco::Bag *> &bag_ctx = context->bag_ctx();
  WeightContext &weight_ctx = context->weight_ctx();

  assert(layer.bottom().size() == 1);
  assert(layer.top().size() == 1);

  assert(layer.has_convolution_param());
  const auto &param = layer.convolution_param();

  ConvolutionSpec spec{param};
  {
    const auto ifm_name = layer.bottom(0);
    const auto ifm_shape = shape_ctx.at(ifm_name);
    spec.ifm_shape(ifm_shape);
  }

  // NOTE The current implementation focuses on 2D convolution
  // TODO Support general ND convolution
  assert(spec.num_batch_axes() == 1);
  assert(spec.num_spatial_axes() == 2);

  // Create an object for an input feature map
  const auto ifm_name = layer.bottom(0);
  const auto ifm_shape = shape_ctx.at(ifm_name);
  auto ifm_bag = bag_ctx.at(ifm_name);
  auto ifm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ifm_obj->bag(ifm_bag);
  ifm_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ifm_shape)));

  // Create an object for an output feature map
  const auto ofm_name = layer.top(0);
  const auto ofm_shape = spec.ofm_shape();
  auto ofm_bag = module->entity()->bag()->create(num_elements(ofm_shape));
  auto ofm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ofm_shape)));

  // Create an object for kernel
  using namespace coco::KernelLayouts;

  const auto ker_shape = spec.ker_shape();
  auto ker_bag = module->entity()->bag()->create(num_elements(ker_shape));
  auto ker_obj = module->entity()->object()->create<coco::KernelObject>();

  ker_obj->bag(ker_bag);
  ker_obj->layout(NCHW::create(as_kernel_shape(ker_shape)));

  // Create a kernel overlay for the kernel object
  data->f32()->allocate(ker_bag);

  // Initialize the kernel overlay
  assert(weight_ctx.blob_count(layer.name()) >= 1);
  auto ker_blob = weight_ctx.blob_get(layer.name(), 0);

  assert(ker_shape == caffeimport::as_tensor_shape(ker_blob));

  auto ker_dst = data->f32()->access(ker_obj);
  auto ker_src = kernel::OverlayFactory<float, kernel::NCHWLayout>::make(
    ker_obj->shape(), ker_blob->mutable_data()->begin());

  for (uint32_t n = 0; n < ker_obj->shape().count(); ++n)
  {
    for (uint32_t ch = 0; ch < ker_obj->shape().depth(); ++ch)
    {
      for (uint32_t row = 0; row < ker_obj->shape().height(); ++row)
      {
        for (uint32_t col = 0; col < ker_obj->shape().width(); ++col)
        {
          ker_dst->at(n, ch, row, col) = ker_src.at(n, ch, row, col);
        }
      }
    }
  }

  // Create a Load op
  auto load = op_builder(module).load(ifm_obj).pop();

  // Create a Conv2D op
  auto op = module->entity()->op()->create<coco::Conv2D>();

  op->group(spec.group());

  op->ker(ker_obj);
  op->stride()->vertical(spec.stride(0));
  op->stride()->horizontal(spec.stride(1));

  op->pad()->top(spec.pad(0));
  op->pad()->bottom(spec.pad(0));
  op->pad()->left(spec.pad(1));
  op->pad()->right(spec.pad(1));

  op->arg(load);

  // Create an Eval instruction
  auto ins = instr_builder(module).eval(ofm_obj, op);

  // Append the instruction to the block
  blk->instr()->append(ins);

  //
  // coco IR allows Conv2D fused with Add, but the current implementation of enco backend
  // is unable to process such a tree.
  //
  // As a workaround, caffe frontend constructs a instruction for Conv2D and Add.
  //
  if (param.bias_term())
  {
    assert(weight_ctx.blob_count(layer.name()) >= 2);

    // Create Bag & Object
    auto bias_bag = module->entity()->bag()->create(ker_shape.dim(0));
    auto bias_obj = module->entity()->object()->create<coco::FeatureObject>();

    bias_obj->bag(bias_bag);
    bias_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

    auto added_bag = module->entity()->bag()->create(num_elements(ofm_shape));
    auto added_obj = module->entity()->object()->create<coco::FeatureObject>();

    added_obj->bag(added_bag);
    added_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ofm_shape)));

    // Create Op
    auto bias_add = op_builder(module).load(bias_obj).load(ofm_obj).add().pop();

    // Create Instr
    auto bias_add_ins = instr_builder(module).eval(added_obj, bias_add);

    // Append the instruction
    blk->instr()->append(bias_add_ins);

    // Fill bias data
    data->f32()->allocate(bias_bag);

    auto bias_span = data->f32()->weight(bias_bag);
    auto bias_blob = weight_ctx.blob_get(layer.name(), 1);

    for (uint32_t ch = 0; ch < ker_obj->shape().count(); ++ch)
    {
      bias_span[ch] = bias_blob->data(ch);
    }

    // Update output
    ofm_bag = added_bag;
  }

  // Update bag and shape context
  bag_ctx[ofm_name] = ofm_bag;
  shape_ctx[ofm_name] = ofm_shape;
}

} // namespace caffeimport
