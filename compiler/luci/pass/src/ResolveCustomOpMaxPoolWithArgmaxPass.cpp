/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ResolveCustomOpMaxPoolWithArgmaxPass.h"

#include "flatbuffers/flexbuffers.h"
#include <loco/IR/DataTypeTraits.h>

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <loco.h>
#include <oops/InternalExn.h>

namespace
{

template <typename T> std::vector<T> to_vector(const flexbuffers::TypedVector &typed_vec)
{
  std::vector<T> answer(typed_vec.size());

  for (uint32_t i = 0; i < answer.size(); ++i)
  {
    answer[i] = typed_vec[i].As<T>();
  }

  return answer;
}

void define_max_pool_shape(const luci::CircleNode *input, luci::CircleMaxPool2D *maxpool)
{
  auto const padding = maxpool->padding();
  auto const stride = maxpool->stride();
  auto const filter = maxpool->filter();

  assert(input->rank() == 4);
  maxpool->rank(input->rank());

  // batch and depth is same
  maxpool->dim(0).set(input->dim(0).value());
  maxpool->dim(3).set(input->dim(3).value());

  // For VALID padding
  auto padded_height = input->dim(1).value();
  auto padded_width = input->dim(2).value();

  assert(padding != luci::Padding::UNDEFINED);

  if (padding == luci::Padding::SAME)
  {
    padded_height += filter->h() - 1;
    padded_width += filter->w() - 1;
  }

  auto const output_height = (padded_height - filter->h()) / stride->h() + 1;
  auto const output_width = (padded_width - filter->w()) / stride->w() + 1;

  maxpool->dim(1).set(output_height);
  maxpool->dim(2).set(output_width);

  maxpool->shape_status(luci::ShapeStatus::VALID);
}

void copy_shape(const luci::CircleNode *src, luci::CircleNode *dst)
{
  assert(src->shape_status() == luci::ShapeStatus::VALID);

  dst->rank(src->rank());
  for (uint32_t i = 0; i < dst->rank(); ++i)
  {
    dst->dim(i).set(src->dim(i).value());
  }

  dst->shape_status(luci::ShapeStatus::VALID);
}

luci::Padding string_to_padding(const std::string &pad_str)
{
  if (pad_str == "VALID")
    return luci::Padding::VALID;
  if (pad_str == "SAME")
    return luci::Padding::SAME;

  return luci::Padding::UNDEFINED;
}

template <loco::DataType DT> void fill_conv_weights(luci::CircleConst *weights)
{
  assert(weights->rank() == 4);

  auto const kn = weights->dim(0).value();
  auto const kh = weights->dim(1).value();
  auto const kw = weights->dim(2).value();

  auto elements_size = kn * kh * kw * 1;
  weights->size<DT>(elements_size);

  for (uint32_t b = 0; b < kn; ++b)
  {
    for (uint32_t y = 0; y < kh; ++y)
    {
      for (uint32_t x = 0; x < kw; ++x)
      {
        auto const idx = (b * kh + y) * kw + x;
        weights->at<DT>(idx) = (y * kw + x == b) ? 1 : 0;
      }
    }
  }
}

luci::CircleConst *create_conv_filter(luci::CircleConv2D *conv, const uint32_t kh,
                                      const uint32_t kw, const uint32_t kn)
{
  auto weights = conv->graph()->nodes()->create<luci::CircleConst>();

  assert(conv->rank() == 4);
  weights->rank(conv->rank());
  weights->dim(0).set(kn);
  weights->dim(1).set(kh);
  weights->dim(2).set(kw);
  weights->dim(3).set(1);

  weights->dtype(conv->dtype());

  switch (weights->dtype())
  {
    case loco::DataType::FLOAT32:
      fill_conv_weights<loco::DataType::FLOAT32>(weights);
      break;
    case loco::DataType::U8:
      fill_conv_weights<loco::DataType::U8>(weights);
      break;
    case loco::DataType::S16:
      fill_conv_weights<loco::DataType::S16>(weights);
      break;
    default:
      // TODO support other data types
      INTERNAL_EXN("Datatype is not supported, yet!");
  }

  return weights;
}

template <loco::DataType DT> void fill_zero_bias(luci::CircleConst *bias)
{
  assert(bias->rank() == 1);
  auto const depth = bias->dim(0).value();

  bias->size<DT>(depth);

  for (uint32_t i = 0; i < depth; ++i)
  {
    bias->at<DT>(i) = 0;
  }
}

luci::CircleConst *create_zero_bias(loco::Graph *graph, const luci::CircleConv2D *conv)
{
  auto bias = graph->nodes()->create<luci::CircleConst>();

  bias->dtype(conv->dtype());
  bias->rank(1);

  assert(conv->rank() == 4);

  bias->dim(0).set(conv->dim(0).value());

  switch (bias->dtype())
  {
    case loco::DataType::FLOAT32:
      fill_zero_bias<loco::DataType::FLOAT32>(bias);
      break;
    case loco::DataType::U8:
      fill_zero_bias<loco::DataType::S32>(bias);
      break;
    case loco::DataType::S16:
      fill_zero_bias<loco::DataType::S64>(bias);
      break;
    default:
      // TODO support other data types
      INTERNAL_EXN("Datatype is not supported, yet!");
  }

  return bias;
}

template <loco::DataType DT, typename Numeric>
luci::CircleConst *create_scalar(loco::Graph *graph, Numeric value)
{
  auto scalar = graph->nodes()->create<luci::CircleConst>();

  scalar->dtype(DT);

  scalar->rank(0);
  scalar->size<DT>(1);
  scalar->shape_status(luci::ShapeStatus::VALID);

  scalar->scalar<DT>() = value;

  return scalar;
}

template <loco::DataType DT>
void fill_coords_addition(luci::Padding padding, const luci::Stride &stride,
                          const luci::Filter &filter, uint32_t, uint32_t input_width,
                          luci::CircleConst *cords)
{
  auto const output_height = static_cast<int32_t>(cords->dim(1).value());
  auto const output_width = static_cast<int32_t>(cords->dim(2).value());

  {
    auto const element_counts = 1 * output_height * output_width * 1;
    cords->size<DT>(element_counts);
  }

  assert(padding != luci::Padding::UNDEFINED);

  // For VALID padding:
  int32_t start_y = 0;
  int32_t start_x = 0;

  // For SAME padding:
  if (padding == luci::Padding::SAME)
  {
    start_y -= static_cast<int32_t>(filter.h() - 1) / 2;
    start_x -= static_cast<int32_t>(filter.w() - 1) / 2;
  }

  auto const step_y = static_cast<int32_t>(stride.h());
  auto const step_x = static_cast<int32_t>(stride.w());

  for (int32_t y_o = 0, y_i = start_y; y_o < output_height; ++y_o, y_i += step_y)
  {
    for (int32_t x_o = 0, x_i = start_y; x_o < output_width; ++x_o, x_i += step_x)
    {
      auto const output_idx = y_o * output_width + x_o;
      auto const input_idx = y_i * input_width + x_i;

      cords->at<DT>(output_idx) = input_idx;
    }
  }
}

luci::CircleConst *create_coords_addition(luci::Padding padding, const luci::Stride &stride,
                                          const luci::Filter &filter, uint32_t input_height,
                                          uint32_t input_width, luci::CircleNode *pooled)
{
  assert(pooled->rank() == 4 && pooled->dim(3) == 1 && pooled->dim(0) == 1);

  auto cords = pooled->graph()->nodes()->create<luci::CircleConst>();

  cords->dtype(pooled->dtype());
  copy_shape(pooled, cords);

  switch (cords->dtype())
  {
    case loco::DataType::FLOAT32:
      fill_coords_addition<loco::DataType::FLOAT32>(padding, stride, filter, input_height,
                                                    input_width, cords);
      break;
    case loco::DataType::U8:
      fill_coords_addition<loco::DataType::U8>(padding, stride, filter, input_height, input_width,
                                               cords);
      break;
    case loco::DataType::S16:
      fill_coords_addition<loco::DataType::S16>(padding, stride, filter, input_height, input_width,
                                                cords);
      break;
    default:
      // TODO support other data types
      INTERNAL_EXN("Datatype is not supported, yet!");
  }

  return cords;
}

bool resolve_max_pool_with_argmax(luci::CircleCustom *cop)
{
#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;

  auto const graph = cop->graph();
  const std::vector<uint8_t> custom_options = cop->custom_options();
  auto map = flexbuffers::GetRoot(custom_options).AsMap();

  auto const origin = luci::get_origin(cop);
  auto const name = cop->name();
  assert(name.length() > 0);

  // Define params
  // Note: Only `Targmax` equal to DT_INT64 is supported by tflite converter
  // Note: Only `data_format` equal to "NHWC" is supported by tflite converter
  // TODO add support of `include_batch_in_index` param
  auto ksize_param = to_vector<uint32_t>(map["ksize"].AsTypedVector());
  auto strides_param = to_vector<uint32_t>(map["strides"].AsTypedVector());
  auto padding_param = map["padding"].As<std::string>();

  // input node
  auto const input = loco::must_cast<luci::CircleNode *>(cop->inputs(0));
  CHECK_OR_FALSE(input->rank() == 4);

  // TODO support batch size > 1 and `include_batch_in_index` option
  CHECK_OR_FALSE(input->dim(0).value() == 1);

  // Depth dimension is 3 because NHWC format is used
  auto const depth_dimension = 3;

  auto const input_depth = input->dim(depth_dimension).value();
  auto const input_height = input->dim(1).value();
  auto const input_width = input->dim(2).value();

  // outputs
  auto const outputs = loco::succs(cop);
  assert(outputs.size() == cop->numOutputs());
  CHECK_OR_FALSE(outputs.size() == 2);

  auto output0 = loco::must_cast<luci::CircleCustomOut *>(*outputs.begin());
  auto output1 = loco::must_cast<luci::CircleCustomOut *>(*(++outputs.begin()));

  if (output0->index() > output1->index())
    std::swap(output0, output1);

  // From TF documentation: output of maxpool must has same type as input
  assert(output0->dtype() == input->dtype());
  assert(output1->dtype() == loco::DataType::S64);

  assert(ksize_param.size() == input->rank());
  assert(strides_param.size() == input->rank());

  // Batch size and depth of ksize more than 1 is not supported.
  CHECK_OR_FALSE(ksize_param[0] == 1 && ksize_param[3] == 1);
  CHECK_OR_FALSE(strides_param[0] == 1 && strides_param[3] == 1);

  // define Padding
  auto padding = string_to_padding(padding_param);

  // define Filter
  luci::Filter filter;
  filter.h(ksize_param[1]);
  filter.w(ksize_param[2]);

  // define Stride
  luci::Stride stride;
  stride.h(strides_param[1]);
  stride.w(strides_param[2]);

  // Create MaxPool
  auto maxpool = graph->nodes()->create<luci::CircleMaxPool2D>();
  {
    maxpool->name(name + "/MaxPool2D");
    luci::add_origin(maxpool, origin);

    maxpool->stride()->h(stride.h());
    maxpool->stride()->w(stride.w());

    maxpool->filter()->h(filter.h());
    maxpool->filter()->w(filter.w());

    maxpool->padding(padding);

    maxpool->fusedActivationFunction(luci::FusedActFunc::NONE);

    // dtype of first output of MaxPoolWithArgmax is same as input's
    maxpool->dtype(input->dtype());
    define_max_pool_shape(input, maxpool);

    maxpool->value(input);
  }

  // Create Split
  auto split = graph->nodes()->create<luci::CircleSplit>();
  {
    split->name(name + "/Split");
    luci::add_origin(split, origin);

    // Create split_dim
    auto split_dim = create_scalar<loco::DataType::S32>(graph, depth_dimension);
    {
      split_dim->name(split->name() + "/Dim");
      luci::add_origin(split_dim, origin);
    }

    // Split input feature map to FM's with depth equal to 1
    split->dtype(input->dtype());
    copy_shape(input, split);
    split->num_split(int32_t(input_depth));

    split->split_dim(split_dim);
    split->input(input);
  }

  // Create Concatenation
  auto concat = graph->nodes()->create<luci::CircleConcatenation>(input_depth);
  {
    concat->name(name + "/Concatenation");
    luci::add_origin(concat, origin);

    concat->fusedActivationFunction(luci::FusedActFunc::NONE);

    // Output of branches is FLOAT32
    concat->dtype(loco::DataType::FLOAT32);
    copy_shape(maxpool, concat);

    concat->axis(depth_dimension);
  }

  /**
   * Note: we need define idx from input_tensor of maximum element in MaxPool's sliding window.
   * For this we split input tensor by channels, define idx in sliding window and convert this idx
   * to idx from source input_tensor using FloorDiv, Mul and Add operations with constant tensors.
   */
  // create branches
  for (uint32_t br_n = 0; br_n < input_depth; ++br_n)
  {
    // Create CircleSplitOut
    auto split_out = graph->nodes()->create<luci::CircleSplitOut>();
    {
      split_out->name(name + "/SplitOut_" + std::to_string(br_n));
      luci::add_origin(split_out, origin);

      split_out->index(br_n);
      split_out->dtype(split->dtype());

      // shape of split out equal to shape of split except split_dim, which equal to 1
      copy_shape(split, split_out);
      split_out->dim(depth_dimension).set(1);

      split_out->input(split);
    }

    // Define idx of max element in Window:
    // Create Conv2D
    auto conv = graph->nodes()->create<luci::CircleConv2D>();
    {
      conv->name(name + "/Conv2d");
      luci::add_origin(conv, origin);

      // Padding, Stride and kernel size equal to MaxPool's
      conv->padding(padding);
      conv->stride()->w(stride.w());
      conv->stride()->h(stride.h());

      auto const kh = filter.h();
      auto const kw = filter.w();

      conv->fusedActivationFunction(luci::FusedActFunc::NONE);

      // dilation is default

      conv->dtype(split_out->dtype());

      // shape of Plane is same as MaxPool's, depth is equal to square size of kernel
      copy_shape(maxpool, conv);
      conv->dim(depth_dimension).set(kh * kw);

      // use no bias
      auto bias = create_zero_bias(graph, conv);
      {
        bias->name(conv->name() + "/Bias");
        luci::add_origin(bias, origin);
      }

      // create filter
      // TODO make shared
      auto weights = create_conv_filter(conv, kh, kw, kh * kw);
      {
        weights->name(conv->name() + "/Weights");
        luci::add_origin(weights, origin);
      }

      conv->bias(bias);
      conv->filter(weights);
      conv->input(split_out);
    }

    // Create ArgMax
    auto argmax = graph->nodes()->create<luci::CircleArgMax>();
    {
      argmax->name(name + "/ArgMax");
      add_origin(argmax, origin);

      argmax->output_type(loco::DataType::S32);
      argmax->dtype(argmax->output_type());

      // argmax returns tensor of max elements by depth axis
      copy_shape(conv, argmax);
      argmax->dim(depth_dimension).set(1);

      // Create argmax_dim
      auto argmax_dim = create_scalar<loco::DataType::S32>(graph, depth_dimension);
      {
        argmax_dim->name(argmax->name() + "/Dimension");
        luci::add_origin(argmax_dim, origin);
      }

      argmax->dimension(argmax_dim);
      argmax->input(conv);
    }

    // Create Cast to use float32 instead int32
    auto argmax_cast = graph->nodes()->create<luci::CircleCast>();
    {
      argmax_cast->name(argmax->name() + "/Cast");
      luci::add_origin(argmax_cast, origin);

      argmax_cast->in_data_type(argmax->dtype());
      argmax_cast->out_data_type(loco::DataType::FLOAT32);
      argmax_cast->dtype(argmax_cast->out_data_type());

      copy_shape(argmax, argmax_cast);

      argmax_cast->x(argmax);
    }

    // Create Div through Mul
    auto div = graph->nodes()->create<luci::CircleMul>();
    {
      div->name(name + "/Div");
      luci::add_origin(div, origin);

      div->fusedActivationFunction(luci::FusedActFunc::NONE);

      div->dtype(argmax_cast->dtype());
      copy_shape(argmax_cast, div);

      auto divider = create_scalar<loco::DataType::FLOAT32>(graph, 1.0f / filter.w());
      {
        divider->name(div->name() + "/Divider");
        luci::add_origin(divider, origin);
      }

      div->x(argmax_cast);
      div->y(divider);
    }

    auto floor = graph->nodes()->create<luci::CircleFloor>();
    {
      floor->name(name + "/Floor");
      luci::add_origin(floor, origin);

      floor->dtype(div->dtype());
      copy_shape(div, floor);

      floor->x(div);
    }

    auto mod = graph->nodes()->create<luci::CircleAdd>();
    {
      mod->name(name + "/Mod");
      luci::add_origin(mod, origin);

      mod->fusedActivationFunction(luci::FusedActFunc::NONE);

      mod->dtype(floor->dtype());
      copy_shape(floor, mod);

      auto neg = graph->nodes()->create<luci::CircleNeg>();
      {
        neg->name(mod->name() + "/Neg");
        luci::add_origin(neg, origin);

        neg->dtype(mod->dtype());
        copy_shape(floor, mod);

        auto mul = graph->nodes()->create<luci::CircleMul>();
        {
          mul->name(neg->name() + "/Mul");
          luci::add_origin(mul, origin);

          mul->fusedActivationFunction(luci::FusedActFunc::NONE);

          mul->dtype(floor->dtype());
          copy_shape(floor, mul);

          auto multipler = create_scalar<loco::DataType::FLOAT32>(graph, filter.w());
          {
            multipler->name(mul->name() + "/Multipler");
            luci::add_origin(multipler, origin);
          }

          mul->x(floor);
          mul->y(multipler);
        }

        neg->x(mul);
      }

      mod->x(argmax_cast);
      mod->y(neg);
    }

    // Define idx of max element in Plane
    auto const window_y_coord = floor;
    auto const window_x_coord = mod;

    // Create Const
    // This tensor contains coords of left top corners for each window from input tensor
    auto cords =
      create_coords_addition(padding, stride, filter, input_height, input_width, argmax_cast);
    {
      cords->name(name + "/Const");
      luci::add_origin(cords, origin);
    }

    // Create Add
    auto add = graph->nodes()->create<luci::CircleAdd>();
    {
      add->name(name + "/Add");
      luci::add_origin(add, origin);

      add->fusedActivationFunction(luci::FusedActFunc::NONE);

      add->dtype(cords->dtype());
      copy_shape(cords, add);

      auto addition = graph->nodes()->create<luci::CircleAdd>();
      {
        addition->name(add->name() + "/Add");
        luci::add_origin(addition, origin);

        addition->fusedActivationFunction(luci::FusedActFunc::NONE);

        addition->dtype(window_x_coord->dtype());
        copy_shape(window_x_coord, addition);

        auto y_addition = graph->nodes()->create<luci::CircleMul>();
        {
          y_addition->name(addition->name() + "/Mul");
          luci::add_origin(y_addition, origin);

          y_addition->fusedActivationFunction(luci::FusedActFunc::NONE);

          y_addition->dtype(window_y_coord->dtype());
          copy_shape(window_y_coord, add);

          auto width_scalar = create_scalar<loco::DataType::FLOAT32>(graph, input_width);
          {
            width_scalar->name(y_addition->name() + "/Const");
            luci::add_origin(width_scalar, origin);
          }

          y_addition->x(window_y_coord);
          y_addition->y(width_scalar);
        }

        addition->x(window_x_coord);
        addition->y(y_addition);
      }

      add->x(addition);
      add->y(cords);
    }

    // Define volume coords
    // Create Mul
    auto mul = graph->nodes()->create<luci::CircleMul>();
    {
      mul->name(name + "/Mul");
      luci::add_origin(mul, origin);

      mul->fusedActivationFunction(luci::FusedActFunc::NONE);

      mul->dtype(add->dtype());
      copy_shape(add, mul);

      auto depth_scalar = create_scalar<loco::DataType::FLOAT32>(graph, input_depth);
      {
        depth_scalar->name(mul->name() + "/Const");
        luci::add_origin(depth_scalar, origin);
      }

      mul->x(add);
      mul->y(depth_scalar);
    }

    luci::CircleNode *branch_out = nullptr;

    if (br_n > 0)
    {
      // Create Add
      auto add_ch = graph->nodes()->create<luci::CircleAdd>();
      {
        add_ch->name(name + "/Add_Channel");
        luci::add_origin(add_ch, origin);

        add_ch->fusedActivationFunction(luci::FusedActFunc::NONE);

        add_ch->dtype(mul->dtype());
        copy_shape(mul, add_ch);

        auto channel_scalar = create_scalar<loco::DataType::FLOAT32>(graph, br_n);
        {
          channel_scalar->name(add_ch->name() + "/Const");
          luci::add_origin(channel_scalar, origin);
        }

        add_ch->x(mul);
        add_ch->y(channel_scalar);
      }

      branch_out = add_ch;
    }
    else
    {
      branch_out = mul;
    }

    concat->values(br_n, branch_out);
  }

  // Output of argmax_with_maxpool should be S64
  auto output_cast = graph->nodes()->create<luci::CircleCast>();
  {
    output_cast->name(name + "/Cast");
    luci::add_origin(output_cast, origin);

    output_cast->in_data_type(concat->dtype());
    output_cast->out_data_type(loco::DataType::S64);
    output_cast->dtype(output_cast->out_data_type());

    copy_shape(concat, output_cast);

    output_cast->x(concat);
  }

  // replace old node with new subgraph
  cop->inputs(0, nullptr);
  loco::replace(output0).with(maxpool);
  loco::replace(output1).with(output_cast);

  return true;
}

} // namespace

namespace luci
{

bool ResolveCustomOpMaxPoolWithArgmaxPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (cop->custom_code() != "MaxPoolWithArgmax")
      continue;

    if (!resolve_max_pool_with_argmax(cop))
      continue;

    changed = true;
  }

  return changed;
}

} // namespace luci
