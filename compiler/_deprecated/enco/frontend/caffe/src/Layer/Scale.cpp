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

#include "Scale.h"
#include "IRBuilder.h"

#include <coco/IR/FeatureLayouts.h>

#include <morph/caffe.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::caffe;

namespace caffeimport
{

void ScaleBuilder::build(const ::caffe::LayerParameter &layer, GraphBuilderContext *context) const
{
  coco::Module *module = context->module();
  coco::Data *data = context->data();
  coco::Block *blk = context->block();
  std::map<std::string, tensor::Shape> &shape_ctx = context->shape_ctx();
  std::map<std::string, coco::Bag *> &bag_ctx = context->bag_ctx();
  WeightContext &weight_ctx = context->weight_ctx();

  // TODO Support Scale layer with 2 bottoms
  assert(layer.bottom().size() == 1);
  assert(layer.top().size() == 1);

  assert(layer.has_scale_param());
  const auto &param = layer.scale_param();

  assert(param.axis() == 1);
  assert(!param.has_num_axes());

  assert(weight_ctx.blob_count(layer.name()) >= 1);

  // NOTE The shape of "Scale" output is same as that of its input
  // NOTE The current implementation assumes that input/output is of feature type
  // TODO Support generic tensor arguments
  auto shape = shape_ctx.at(layer.bottom(0));

  coco::Bag *last_bag = bag_ctx.at(layer.bottom(0));

  // Create channel-wise multiplication
  {
    auto in_bag = last_bag;
    auto in_obj = module->entity()->object()->create<coco::FeatureObject>();

    in_obj->bag(in_bag);
    in_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(shape)));

    auto factor_bag = module->entity()->bag()->create(num_elements(shape));
    auto factor_obj = module->entity()->object()->create<coco::FeatureObject>();

    factor_obj->bag(factor_bag);
    factor_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(shape)));

    auto out_bag = module->entity()->bag()->create(num_elements(shape));
    auto out_obj = module->entity()->object()->create<coco::FeatureObject>();

    out_obj->bag(out_bag);
    out_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(shape)));

    auto mul_op = op_builder(module).load(factor_obj).load(in_obj).mul().pop();
    auto mul_ins = instr_builder(module).eval(out_obj, mul_op);

    blk->instr()->append(mul_ins);

    // Fill "factor" data
    {
      data->f32()->allocate(factor_bag);

      auto span = data->f32()->weight(factor_bag);
      auto blob = weight_ctx.blob_get(layer.name(), 0);

      for (uint32_t ch = 0; ch < factor_obj->shape().depth(); ++ch)
      {
        span[ch] = blob->data(ch);
      }
    }

    // Update "last_bag"
    last_bag = out_bag;
  }

  assert(last_bag != nullptr);

  // Create bias addition (as channel-wise addition)
  if (param.bias_term())
  {
    assert(weight_ctx.blob_count(layer.name()) >= 2);

    auto in_bag = last_bag; /* Use the output of the last computation as an input */
    auto in_obj = module->entity()->object()->create<coco::FeatureObject>();

    in_obj->bag(in_bag);
    in_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(shape)));

    auto bias_bag = module->entity()->bag()->create(num_elements(shape));
    auto bias_obj = module->entity()->object()->create<coco::FeatureObject>();

    bias_obj->bag(bias_bag);
    bias_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(shape)));

    auto out_bag = module->entity()->bag()->create(num_elements(shape));
    auto out_obj = module->entity()->object()->create<coco::FeatureObject>();

    out_obj->bag(out_bag);
    out_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(shape)));

    auto add_op = op_builder(module).load(bias_obj).load(in_obj).add().pop();
    auto add_ins = instr_builder(module).eval(out_obj, add_op);

    blk->instr()->append(add_ins);

    // Fill bias data
    {
      data->f32()->allocate(bias_bag);

      auto bias_span = data->f32()->weight(bias_bag);
      auto bias_blob = weight_ctx.blob_get(layer.name(), 1);

      for (uint32_t ch = 0; ch < bias_obj->shape().depth(); ++ch)
      {
        bias_span[ch] = bias_blob->data(ch);
      }
    }

    // Update "last_bag"
    last_bag = out_bag;
  }

  // Update bag and shape context
  {
    const auto &out_name = layer.top(0);
    const auto &out_bag = last_bag;
    const auto &out_shape = shape;

    bag_ctx[out_name] = out_bag;
    shape_ctx[out_name] = out_shape;
  }
}

} // namespace caffeimport
