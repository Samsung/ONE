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

#include "Concatenation.h"
#include "IRBuilder.h"

#include <coco/IR/FeatureLayouts.h>

#include <morph/caffe.h>

#include <cassert>

using namespace nncc::core::ADT;
using namespace morph::caffe;

namespace caffeimport
{

void ConcatBuilder::build(const ::caffe::LayerParameter &layer, GraphBuilderContext *context) const
{
  coco::Module *module = context->module();
  coco::Block *blk = context->block();
  std::map<std::string, tensor::Shape> &shape_ctx = context->shape_ctx();
  std::map<std::string, coco::Bag *> &bag_ctx = context->bag_ctx();

  assert(layer.bottom().size() > 0);
  assert(layer.top().size() == 1);

  // Assume default concat axis
  // - Please refer to http://caffe.berkeleyvision.org/tutorial/layers/concat.html for details
  // TODO Get concat axis from concat param
  assert(!layer.has_concat_param());
  const uint32_t concat_axis = 1;

  // Construct a vector of input objects
  std::vector<coco::FeatureObject *> input_objects;

  for (const auto &input_name : layer.bottom())
  {
    const auto input_shape = as_feature_shape(shape_ctx.at(input_name));

    auto input_bag = bag_ctx.at(input_name);
    auto input_feature = module->entity()->object()->create<coco::FeatureObject>();

    input_feature->bag(input_bag);
    input_feature->layout(coco::FeatureLayouts::BCHW::create(input_shape));

    input_objects.emplace_back(input_feature);
  }

  coco::FeatureObject *last_feature = input_objects.at(0);

  assert(last_feature != nullptr);
  assert(last_feature->bag() != nullptr);

  // Update coco IR
  //
  // Given a sequence of input features %in[0] / %in[1] / ... / %in[N]
  // the below code constructs a sequence of eval instructions
  //  - Load is omitted for simplicity
  //
  // %out[0] = eval(ConcatF(%in[0], %in[1]))
  // %out[1] = eval(ConcatF(%out[0], %in[2]))
  // ...
  // %out[N - 1] = eval(ConcatF(%out[N - 2], %in[N]))
  //
  for (uint32_t n = 1; n < input_objects.size(); ++n)
  {
    auto const left_feature = last_feature;
    auto const left_shape = left_feature->layout()->shape();

    auto right_feature = input_objects.at(n);
    auto right_shape = right_feature->layout()->shape();

    // Batch is not supported, yet
    assert(left_feature->layout()->batch() == 1);
    assert(right_feature->layout()->batch() == 1);

    // Height and Width SHOULD BE IDENTICAL for depth concat
    assert(left_shape.height() == right_shape.height());
    assert(left_shape.width() == right_shape.width());

    const uint32_t C = left_shape.depth() + right_shape.depth();
    const uint32_t H = left_shape.height();
    const uint32_t W = left_shape.width();

    const nncc::core::ADT::feature::Shape out_shape{C, H, W};

    auto out_bag = module->entity()->bag()->create(num_elements(out_shape));
    auto out_feature = module->entity()->object()->create<coco::FeatureObject>();

    out_feature->bag(out_bag);
    out_feature->layout(coco::FeatureLayouts::BCHW::create(out_shape));

    auto left_load = op_builder(module).load(left_feature).pop();
    auto right_load = op_builder(module).load(right_feature).pop();

    auto concat_f = module->entity()->op()->create<coco::ConcatF>();

    concat_f->axis(coco::ConcatF::Axis::Depth);
    concat_f->left(left_load);
    concat_f->right(right_load);

    auto eval = instr_builder(module).eval(out_feature, concat_f);

    // Append the constructed Shuffle instruction
    blk->instr()->append(eval);

    // Update 'last_feature'
    last_feature = out_feature;
  }

  assert(last_feature != nullptr);
  assert(last_feature->bag() != nullptr);

  // Update bag and shape context
  auto const out_name = layer.top(0);
  auto const out_shape = as_tensor_shape(last_feature->layout()->shape());
  auto const out_bag = last_feature->bag();

  bag_ctx[out_name] = out_bag;
  shape_ctx[out_name] = out_shape;
}

} // namespace caffeimport
