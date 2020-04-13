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

#include "BatchNorm.h"
#include "IRBuilder.h"

#include <morph/caffe.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::caffe;

using tensor::num_elements;

namespace caffeimport
{

void BatchNormBuilder::build(const ::caffe::LayerParameter &layer,
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

  assert(layer.has_batch_norm_param());
  const auto &param = layer.batch_norm_param();

  // TODO Support training case
  assert(param.use_global_stats() == true);

  // Create an object for an input feature map
  const auto ifm_name = layer.bottom(0);
  const auto ifm_shape = shape_ctx.at(ifm_name);
  auto ifm_bag = bag_ctx.at(ifm_name);
  auto ifm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ifm_obj->bag(ifm_bag);
  ifm_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ifm_shape)));

  // Create an object for an output feature map
  const auto ofm_name = layer.top(0);
  const auto ofm_shape = ifm_shape;
  auto ofm_bag = module->entity()->bag()->create(num_elements(ofm_shape));
  auto ofm_obj = module->entity()->object()->create<coco::FeatureObject>();

  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ofm_shape)));

  // Create an object for the scaled mean estimates data
  auto mean_bag = module->entity()->bag()->create(ofm_shape.dim(1));
  auto mean_obj = module->entity()->object()->create<coco::FeatureObject>();

  mean_obj->bag(mean_bag);
  mean_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

  // Create an object for the scaled variance estimates data
  auto variance_bag = module->entity()->bag()->create(ofm_shape.dim(1));
  auto variance_obj = module->entity()->object()->create<coco::FeatureObject>();

  variance_obj->bag(variance_bag);
  variance_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

  if (param.use_global_stats())
  {
    // Use the stored mean/variance estimates.
    assert(weight_ctx.blob_count(layer.name()) == 3);

    // Create an object for scale factor data
    auto factor_bag = module->entity()->bag()->create(ofm_shape.dim(1));
    auto factor_obj = module->entity()->object()->create<coco::FeatureObject>();

    factor_obj->bag(factor_bag);
    factor_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

    // Fill "scale factor" data
    {
      data->f32()->allocate(factor_bag);

      auto dst = data->f32()->weight(factor_bag);
      // Calculate scale factor
      auto blob = weight_ctx.blob_get(layer.name(), 2);
      const auto scale_factor = blob->data(0) == 0 ? 0.f : 1 / blob->data(0);

      for (uint32_t ch = 0; ch < factor_obj->shape().depth(); ++ch)
      {
        dst[ch] = scale_factor;
      }
    }

    // Create an object for saved mean data
    auto saved_mean_bag = module->entity()->bag()->create(ofm_shape.dim(1));
    auto saved_mean_obj = module->entity()->object()->create<coco::FeatureObject>();

    saved_mean_obj->bag(saved_mean_bag);
    saved_mean_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

    // Fill "saved mean estimates" data
    {
      data->f32()->allocate(saved_mean_bag);

      auto dst = data->f32()->weight(saved_mean_bag);
      auto blob = weight_ctx.blob_get(layer.name(), 0);

      for (uint32_t ch = 0; ch < saved_mean_obj->shape().depth(); ++ch)
      {
        dst[ch] = blob->data(ch);
      }
    }

    // Multiply scale factor to mean data
    {
      auto mul_op = op_builder(module).load(factor_obj).load(saved_mean_obj).mul().pop();
      auto mul_ins = instr_builder(module).eval(mean_obj, mul_op);

      blk->instr()->append(mul_ins);
    }

    // Create an object for saved variance data
    auto saved_variance_bag = module->entity()->bag()->create(ofm_shape.dim(1));
    auto saved_variance_obj = module->entity()->object()->create<coco::FeatureObject>();

    saved_variance_obj->bag(saved_variance_bag);
    saved_variance_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

    // Fill "saved variance estimates" data
    {
      data->f32()->allocate(saved_variance_bag);

      auto dst = data->f32()->weight(saved_variance_bag);
      auto blob = weight_ctx.blob_get(layer.name(), 1);

      for (uint32_t ch = 0; ch < saved_variance_obj->shape().depth(); ++ch)
      {
        dst[ch] = blob->data(ch);
      }
    }

    // Multiply scale factor to variance data
    {
      auto mul_op = op_builder(module).load(factor_obj).load(saved_variance_obj).mul().pop();
      auto mul_ins = instr_builder(module).eval(variance_obj, mul_op);

      blk->instr()->append(mul_ins);
    }
  }
  else
  {
    // TODO use_global_stats() == false case
  }

  // Create an object for subtraction
  auto sub_bag = module->entity()->bag()->create(num_elements(ofm_shape));
  auto sub_obj = module->entity()->object()->create<coco::FeatureObject>();

  sub_obj->bag(sub_bag);
  sub_obj->layout(coco::FeatureLayouts::BCHW::create(as_feature_shape(ofm_shape)));

  // Subtract mean
  {
    auto sub_op = op_builder(module).load(mean_obj).load(ifm_obj).sub().pop();
    auto sub_ins = instr_builder(module).eval(sub_obj, sub_op);

    blk->instr()->append(sub_ins);
  }

  // Create an object for normalize variance data
  auto norm_bag = module->entity()->bag()->create(ofm_shape.dim(1));
  auto norm_obj = module->entity()->object()->create<coco::FeatureObject>();

  norm_obj->bag(norm_bag);
  norm_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

  // Normalize variance
  {
    // Create an object for epsilon data
    auto eps_bag = module->entity()->bag()->create(ofm_shape.dim(1));
    auto eps_obj = module->entity()->object()->create<coco::FeatureObject>();

    eps_obj->bag(eps_bag);
    eps_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

    // Fill "epsilon" data
    {
      data->f32()->allocate(eps_bag);

      auto dst = data->f32()->weight(eps_bag);
      auto eps = param.eps();

      for (uint32_t ch = 0; ch < eps_obj->shape().depth(); ++ch)
      {
        dst[ch] = eps;
      }
    }

    // Create a temp object
    auto temp_bag = module->entity()->bag()->create(ofm_shape.dim(1));
    auto temp_obj = module->entity()->object()->create<coco::FeatureObject>();

    temp_obj->bag(temp_bag);
    temp_obj->layout(coco::FeatureLayouts::BC::create(as_feature_shape(ofm_shape)));

    // Add epsilon to variance
    {
      auto add_op = op_builder(module).load(variance_obj).load(eps_obj).add().pop();
      auto add_ins = instr_builder(module).eval(temp_obj, add_op);

      blk->instr()->append(add_ins);
    }

    // Sqrt variance
    {
      auto load = op_builder(module).load(temp_obj).pop();
      auto sqrt_op = module->entity()->op()->create<coco::Sqrt>();
      sqrt_op->arg(load);
      auto sqrt_ins = instr_builder(module).eval(norm_obj, sqrt_op);

      blk->instr()->append(sqrt_ins);
    }
  }

  // Replicate variance to input size
  {
    auto div_op = op_builder(module).load(norm_obj).load(sub_obj).div().pop();
    auto div_ins = instr_builder(module).eval(ofm_obj, div_op);

    blk->instr()->append(div_ins);
  }

  // Update bag and shape context
  bag_ctx[ofm_name] = ofm_bag;
  shape_ctx[ofm_name] = ofm_shape;
}

} // namespace caffeimport
