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

luci::Padding string_to_padding(const std::string &pad_str)
{
  if (pad_str == "VALID")
    return luci::Padding::VALID;
  if (pad_str == "SAME")
    return luci::Padding::SAME;

  return luci::Padding::UNDEFINED;
}

template <typename NodeT> void set_stride(NodeT *node, const luci::Stride &stride)
{
  node->stride()->h(stride.h());
  node->stride()->w(stride.w());
}

template <typename NodeT> void set_filter(NodeT *node, const luci::Filter &filter)
{
  node->filter()->h(filter.h());
  node->filter()->w(filter.w());
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

void init_name_and_origin(luci::CircleNode *node, const std::string &name,
                          const std::shared_ptr<luci::CircleNodeOrigin> &origin)
{
  node->name(name);
  luci::add_origin(node, origin);
}

void set_shape_and_dtype_from_node(const luci::CircleNode *src, luci::CircleNode *dst)
{
  copy_shape(src, dst);
  dst->dtype(src->dtype());
}

template <typename NodeT> NodeT *none_act_func(NodeT *node)
{
  node->fusedActivationFunction(luci::FusedActFunc::NONE);
  return node;
}

template <loco::DataType DT> luci::CircleCast *create_cast(luci::CircleNode *input)
{
  auto cast = input->graph()->nodes()->create<luci::CircleCast>();

  cast->in_data_type(input->dtype());
  cast->out_data_type(DT);
  cast->dtype(DT);

  copy_shape(input, cast);

  cast->x(input);

  return cast;
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

  bias->dim(0).set(conv->dim(3).value());

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

  // input node
  auto const input = loco::must_cast<luci::CircleNode *>(cop->inputs(0));
  CHECK_OR_FALSE(input->rank() == 4);

  assert(ksize_param.size() == input->rank());
  assert(strides_param.size() == input->rank());

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
  auto output1 = loco::must_cast<luci::CircleCustomOut *>(*outputs.rbegin());

  // sort outputs
  if (output0->index() > output1->index())
    std::swap(output0, output1);

  // From TF documentation: output of maxpool must has same type as input
  assert(output0->dtype() == input->dtype());
  assert(output1->dtype() == loco::DataType::S64);

  // Create MaxPool
  auto maxpool = none_act_func(graph->nodes()->create<luci::CircleMaxPool2D>());
  {
    init_name_and_origin(maxpool, name + "/MaxPool2D", origin);
    set_shape_and_dtype_from_node(input, maxpool);

    set_stride(maxpool, stride);
    set_filter(maxpool, filter);
    maxpool->padding(padding);

    maxpool->value(input);
  }

  // Create Split
  auto split = graph->nodes()->create<luci::CircleSplit>();
  {
    init_name_and_origin(split, name + "/Split", origin);
    set_shape_and_dtype_from_node(input, split);

    // Create split_dim
    auto split_dim = create_scalar<loco::DataType::S32>(graph, depth_dimension);
    init_name_and_origin(split_dim, split->name() + "/Dim", origin);

    split->num_split(int32_t(input_depth));

    split->split_dim(split_dim);
    split->input(input);
  }

  // Create Concatenation
  auto concat = none_act_func(graph->nodes()->create<luci::CircleConcatenation>(input_depth));
  {
    init_name_and_origin(concat, name + "/Concatenation", origin);

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
    auto const branch_name = name + "/depth_" + std::to_string(br_n);

    // Create CircleSplitOut
    auto split_out = graph->nodes()->create<luci::CircleSplitOut>();
    {
      init_name_and_origin(split_out, branch_name + "/SplitOut", origin);
      set_shape_and_dtype_from_node(split, split_out);

      split_out->index(int32_t(br_n));

      // shape of split out equal to shape of split except split_dim, which equal to 1
      split_out->dim(depth_dimension).set(1);

      split_out->input(split);
    }

    // Define idx of max element in Window:
    // Create Conv2D
    auto conv = none_act_func(graph->nodes()->create<luci::CircleConv2D>());
    {
      init_name_and_origin(conv, branch_name + "/Conv2D", origin);

      // Padding, Stride and kernel size equal to MaxPool's
      set_stride(conv, stride);
      conv->padding(padding);

      // depth of kernel is equal to square size
      auto const kh = filter.h();
      auto const kw = filter.w();
      auto const kd = kh * kw;

      // shape of Plane is same as MaxPool's,
      set_shape_and_dtype_from_node(maxpool, conv);
      conv->dim(depth_dimension).set(kd);

      // use zero bias
      auto bias = create_zero_bias(graph, conv);
      init_name_and_origin(bias, conv->name() + "/Bias", origin);

      // create filter
      // TODO make shared
      auto weights = create_conv_filter(conv, kh, kw, kd);
      init_name_and_origin(weights, conv->name() + "/Weights", origin);

      conv->bias(bias);
      conv->filter(weights);
      conv->input(split_out);
    }

    // Create ArgMax
    auto argmax = graph->nodes()->create<luci::CircleArgMax>();
    {
      init_name_and_origin(argmax, branch_name + "/ArgMax", origin);

      argmax->output_type(loco::DataType::S32);
      argmax->dtype(argmax->output_type());

      // argmax returns tensor of max elements by depth axis
      copy_shape(conv, argmax);
      argmax->dim(depth_dimension).set(1);

      // Create argmax_dim
      auto argmax_dim = create_scalar<loco::DataType::S32>(graph, depth_dimension);
      init_name_and_origin(argmax_dim, argmax->name() + "/Dimension", origin);

      argmax->dimension(argmax_dim);
      argmax->input(conv);
    }

    // Create Cast to use float32 instead int32
    auto argmax_cast = create_cast<loco::DataType::FLOAT32>(argmax);
    ;
    init_name_and_origin(argmax_cast, argmax->name() + "/Cast", origin);

    // Create Div through Mul
    auto div = none_act_func(graph->nodes()->create<luci::CircleMul>());
    {
      init_name_and_origin(div, branch_name + "/Div", origin);
      set_shape_and_dtype_from_node(argmax_cast, div);

      auto divider = create_scalar<loco::DataType::FLOAT32>(graph, 1.0f / filter.w());
      init_name_and_origin(divider, div->name() + "/Divider", origin);

      div->x(argmax_cast);
      div->y(divider);
    }

    auto floor = graph->nodes()->create<luci::CircleFloor>();
    {
      init_name_and_origin(floor, branch_name + "/Floor", origin);
      set_shape_and_dtype_from_node(div, floor);

      floor->x(div);
    }

    auto mod = none_act_func(graph->nodes()->create<luci::CircleAdd>());
    {
      init_name_and_origin(mod, branch_name + "/Mod", origin);
      set_shape_and_dtype_from_node(floor, mod);

      auto neg = graph->nodes()->create<luci::CircleNeg>();
      {
        init_name_and_origin(neg, mod->name() + "/Neg", origin);
        set_shape_and_dtype_from_node(floor, neg);

        auto mul = none_act_func(graph->nodes()->create<luci::CircleMul>());
        {
          init_name_and_origin(mul, neg->name() + "/Neg", origin);
          set_shape_and_dtype_from_node(floor, mul);

          auto multipler = create_scalar<loco::DataType::FLOAT32>(graph, filter.w());
          init_name_and_origin(multipler, mul->name() + "/Multipler", origin);

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
    init_name_and_origin(cords, branch_name + "/Const", origin);

    // Create Add
    auto add = none_act_func(graph->nodes()->create<luci::CircleAdd>());
    {
      init_name_and_origin(add, branch_name + "/Add", origin);
      set_shape_and_dtype_from_node(cords, add);

      auto addition = none_act_func(graph->nodes()->create<luci::CircleAdd>());
      {
        init_name_and_origin(addition, add->name() + "/Add", origin);
        set_shape_and_dtype_from_node(window_x_coord, addition);

        auto y_addition = none_act_func(graph->nodes()->create<luci::CircleMul>());
        {
          init_name_and_origin(y_addition, addition->name() + "/Mul", origin);
          set_shape_and_dtype_from_node(window_y_coord, y_addition);

          auto width_scalar = create_scalar<loco::DataType::FLOAT32>(graph, input_width);
          init_name_and_origin(width_scalar, y_addition->name() + "/Const", origin);

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
    auto mul = none_act_func(graph->nodes()->create<luci::CircleMul>());
    {
      init_name_and_origin(mul, branch_name + "/Mul", origin);
      set_shape_and_dtype_from_node(add, mul);

      auto depth_scalar = create_scalar<loco::DataType::FLOAT32>(graph, input_depth);
      init_name_and_origin(depth_scalar, mul->name() + "/Const", origin);

      mul->x(add);
      mul->y(depth_scalar);
    }

    luci::CircleNode *branch_out = nullptr;

    if (br_n > 0)
    {
      // Create Add
      auto add_ch = none_act_func(graph->nodes()->create<luci::CircleAdd>());
      {
        init_name_and_origin(add_ch, branch_name + "/Add_Channel", origin);
        set_shape_and_dtype_from_node(mul, add_ch);

        auto channel_scalar = create_scalar<loco::DataType::FLOAT32>(graph, br_n);
        init_name_and_origin(channel_scalar, add_ch->name() + "/Const", origin);

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
  auto output_cast = create_cast<loco::DataType::S64>(concat);
  init_name_and_origin(output_cast, name + "/Cast", origin);

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
