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
#include "GraphBuilder.h"

#include <coco/IR/Module.h>
#include <coco/IR/FeatureLayouts.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <array>
#include <cassert>

using namespace nncc::core::ADT;

namespace
{

/**
 * @brief Convert a numeric tensor axis as a ConcatF FeatureAxis value
 */
coco::ConcatF::Axis as_ConcatF_axis(uint32_t axis)
{
  // NOTE The feature map (in TensorFlow) is a rank-4 (NHWC) tensor
  assert(axis < 4);

  coco::ConcatF::Axis res = coco::ConcatF::Axis::Unknown;

  switch (axis)
  {
    case 0:
      res = coco::ConcatF::Axis::Batch;
      break;
    case 1:
      res = coco::ConcatF::Axis::Height;
      break;
    case 2:
      res = coco::ConcatF::Axis::Width;
      break;
    case 3:
      res = coco::ConcatF::Axis::Depth;
      break;
    default:
      break;
  }

  return res;
}

/**
 * @brief Convert a coco FeatureShape as an array of 'uint32_t' values
 */
std::array<uint32_t, 4> as_dims(const coco::FeatureShape &shape)
{
  std::array<uint32_t, 4> res;

  res[0] = shape.batch();
  res[1] = shape.height();
  res[2] = shape.width();
  res[3] = shape.depth();

  return res;
}

/**
 * @brief Convert a tensor shape as a coco FeatureShape
 */
coco::FeatureShape as_feature_shape(const tensor::Shape &shape)
{
  assert(shape.rank() == 4);

  auto const B = shape.dim(0);
  auto const C = shape.dim(3);
  auto const H = shape.dim(1);
  auto const W = shape.dim(2);

  return coco::FeatureShape{B, C, H, W};
}

} // namespace

namespace tflimport
{

void ConcatenationGraphBuilder::build(const tflite::Operator *op,
                                      GraphBuilderContext *context) const
{
  assert(context != nullptr);

  coco::Module *m = context->m();
  coco::Data *d = context->d();
  coco::Block *blk = context->block();
  TensorContext &tensor_context = context->tensor();
  TensorBags &bags = context->bags();
  IndexVector opinputs = as_index_vector(op->inputs());
  IndexVector opoutputs = as_index_vector(op->outputs());

  // these are fixed in tflite
  // input index 0 ~ N : any number of input features
  // output index 0 : one output feature
  assert(opinputs.size() > 0);
  assert(opoutputs.size() == 1);

  // Default parameter values are referenced from schema_generated.h
  int32_t concat_axis = 0;
  tflite::ActivationFunctionType activation = tflite::ActivationFunctionType_NONE;

  if (auto *concatenation_params = op->builtin_options_as_ConcatenationOptions())
  {
    activation = concatenation_params->fused_activation_function();
    concat_axis = concatenation_params->axis();

    const int32_t rank = static_cast<int32_t>(tensor_context.shape(opinputs.at(0)).rank());
    if (concat_axis < 0)
    {
      concat_axis += rank;
    }
    assert(concat_axis >= 0);
    assert(concat_axis < rank);
  }
  assert(as_ConcatF_axis(concat_axis) != coco::ConcatF::Axis::Unknown);
  assert(activation == tflite::ActivationFunctionType_NONE);

  // Construct a vector of input objects
  std::vector<coco::FeatureObject *> input_objects;

  for (auto &input_index : opinputs)
  {
    const tensor::Shape &input_shape = tensor_context.shape(input_index);
    coco::FeatureObject *input_obj = m->entity()->object()->create<coco::FeatureObject>();
    coco::Bag *input_bag = bags.bag(input_index);
    input_obj->bag(input_bag);
    input_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(input_shape)));

    input_objects.emplace_back(input_obj);
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
  // %tmp = eval(ConcatF(%in[0], %in[1]))
  // %tmp = eval(ConcatF(%tmp, %in[2]))
  // ...
  // %tmp = eval(ConcatF(%tmp, %in[N]))
  // %out[0] = copy(%tmp)
  //
  for (uint32_t n = 1; n < input_objects.size(); ++n)
  {
    auto const left_feature = last_feature;
    auto const left_shape = left_feature->layout()->shape();

    auto right_feature = input_objects.at(n);
    auto right_shape = right_feature->layout()->shape();

    // Compute output dimensionalities
    auto compute_out_dims = [&left_shape, &right_shape, concat_axis](void) {
      std::array<uint32_t, 4> out_dims;

      const auto left_dims = as_dims(left_shape);
      const auto right_dims = as_dims(right_shape);

      for (uint32_t axis = 0; axis < 4 /* FEATURE MAP RANK */; ++axis)
      {
        // The dimensionality of all the axises except 'concat' axis SHOULD BE INDETICAL
        assert((concat_axis == axis) || (left_dims[axis] == right_dims[axis]));

        out_dims[axis] = left_dims[axis];
        if (axis == concat_axis)
        {
          out_dims[axis] += right_dims[axis];
        }
      }

      return out_dims;
    };

    const auto out_dims = compute_out_dims();

    const uint32_t B = out_dims[0 /* BATCH */];
    const uint32_t C = out_dims[3 /* DEPTH */];
    const uint32_t H = out_dims[1 /* HEIGHT */];
    const uint32_t W = out_dims[2 /* WIDTH */];

    const coco::FeatureShape out_shape{B, C, H, W};

    auto out_bag = m->entity()->bag()->create(B * num_elements(out_shape));
    auto out_feature = m->entity()->object()->create<coco::FeatureObject>();

    out_feature->bag(out_bag);
    out_feature->layout(coco::FeatureLayouts::BHWC::create(out_shape));

    auto left_load = op_builder(m).load(left_feature).pop();
    auto right_load = op_builder(m).load(right_feature).pop();

    auto concat_f = m->entity()->op()->create<coco::ConcatF>();

    concat_f->axis(as_ConcatF_axis(concat_axis));
    concat_f->left(left_load);
    concat_f->right(right_load);

    auto eval = instr_builder(m).eval(out_feature, concat_f);

    // Append the constructed Shuffle instruction
    blk->instr()->append(eval);

    // Update 'last_feature'
    last_feature = out_feature;
  }

  // Insert copy instruction from last_feature to output operand
  int const ofm_idx = opoutputs.at(0);
  auto const ofm_shape = tensor_context.shape(ofm_idx);

  auto ofm_bag = bags.bag(ofm_idx);
  auto ofm_obj = m->entity()->object()->create<coco::FeatureObject>();

  ofm_obj->bag(ofm_bag);
  ofm_obj->layout(coco::FeatureLayouts::BHWC::create(as_feature_shape(ofm_shape)));

  // Create a Copy instruction from last into ofm
  auto copy_ins = instr_builder(m).copy(ofm_obj, last_feature);

  // Append the instruction
  blk->instr()->append(copy_ins);
}

} // namespace tflimport
