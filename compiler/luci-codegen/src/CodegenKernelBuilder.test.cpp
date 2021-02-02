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

#include <luci/IR/Nodes/CircleInput.h>

#include "luci/IR/CircleNodes.h"

#include "gtest/gtest.h"

#include "CodegenKernelBuilder.h"

#include "Halide.h"

#include <type_traits>

template <loco::DataType DType>
void constructBasicNode(luci::CircleNode &node, const std::vector<int> &dims)
{
  node.dtype(DType);
  node.rank(dims.size());
  for (int i = 0; i < dims.size(); ++i)
  {
    node.dim(i).set(dims[i]);
  }
  node.shape_status(luci::ShapeStatus::VALID);
}

template<typename T>
std::vector<T> reverse_vector(const std::vector<T> &v)
{
  std::vector<T> rev;
  std::reverse_copy(v.begin(), v.end(), std::back_insert_iterator<std::vector<T>>(rev));
  return rev;
}

// todo are luci-interpreter comparators better?
//template <typename T, bool>
//void compare_arrays(T *ref_data, T *result, int size);
template <typename T>
static void compare_arrays(const T *ref_data, const T *result, int size)
{
  for (int i = 0; i < size; ++i)
    ASSERT_EQ(ref_data[i], result[i]);
}

template <>
void compare_arrays<float>(const float *ref_data, const float *result, int size)
{
  for (int i = 0; i < size; ++i)
    ASSERT_FLOAT_EQ(ref_data[i], result[i]) << " index " << i;
}

template <>
void compare_arrays<double>(const double *ref_data, const double *result, int size)
{
  for (int i = 0; i < size; ++i)
    ASSERT_DOUBLE_EQ(ref_data[i], result[i]);
}

// a little hack to simplify definitions of functions
// othrwise it is hard to read code
// todo maybe add reverse c++ types -> loco types transformation?
template<loco::DataType DType>
using DataVector = std::vector<typename loco::DataTypeImpl<DType>::Type>;

using Shape = std::vector<int>;

template<loco::DataType DType>
using Type = typename loco::DataTypeImpl<DType>::Type;

template <typename LuciOp, loco::DataType DType>
void test_binary_op(const Shape &x_shape, const Shape &y_shape, const Shape &out_shape, DataVector<DType> x_data, DataVector<DType> y_data, const DataVector<DType> &ref_out_data)
{
  // construct test graph
  luci::CircleInput input_x;
  constructBasicNode<DType>(input_x, x_shape);
  luci::CircleInput input_y;
  constructBasicNode<DType>(input_y, y_shape);

  LuciOp op;
  constructBasicNode<DType>(op, out_shape);
  op.x(&input_x);
  op.y(&input_y);

  luci::CircleOutput output_node;
  constructBasicNode<DType>(output_node, out_shape);
  output_node.from(&op);

  ASSERT_TRUE(luci_codegen::CodegenKernelBuilder::is_supported(&op));

  luci_codegen::SubgraphContext subgraph;
  subgraph.add_node(&op);
  subgraph.finish_construction();

  luci_codegen::CodegenKernelBuilder builder(subgraph);
  builder.process();

  Halide::Buffer<Type<DType>> x(x_data.data(), reverse_vector(x_shape));
  Halide::Buffer<Type<DType>> y(y_data.data(), reverse_vector(y_shape));
  Halide::Buffer<Type<DType>> res(reverse_vector(out_shape));

  Halide::ImageParam input_param_x = subgraph.get_inputs()[0].second;
  Halide::ImageParam input_param_y = subgraph.get_inputs()[1].second;

  Halide::ParamMap params;
  params.set(input_param_x, x);
  params.set(input_param_y, y);

  Halide::Func target_func = subgraph.get_outputs()[0].second;
  target_func.realize(res, Halide::Target(), params);

  compare_arrays<Type<DType>>(ref_out_data.data(), res.data(), ref_out_data.size());
}

template <typename LuciOp, loco::DataType DType>
void test_unary_op(const Shape &in_out_shape, DataVector<DType> in_data, const DataVector<DType> &ref_out_data)
{
  // construct test graph
  luci::CircleInput input_node;
  constructBasicNode<DType>(input_node, in_out_shape);

  LuciOp op_node;
  constructBasicNode<DType>(op_node, in_out_shape);
  op_node.x(&input_node);

  luci::CircleOutput output_node;
  constructBasicNode<DType>(output_node, in_out_shape);
  output_node.from(&op_node);

  ASSERT_TRUE(luci_codegen::CodegenKernelBuilder::is_supported(&op_node));

  luci_codegen::SubgraphContext subgraph;
  subgraph.add_node(&op_node);
  subgraph.finish_construction();

  luci_codegen::CodegenKernelBuilder builder(subgraph);
  builder.process();

  Halide::Buffer<Type<DType>> input_buffer(in_data.data(), reverse_vector(in_out_shape));
  Halide::Buffer<Type<DType>> res(reverse_vector(in_out_shape));

  Halide::ImageParam input_param = subgraph.get_inputs()[0].second;

  Halide::ParamMap params;
  params.set(input_param, input_buffer);

  Halide::Func target_func = subgraph.get_outputs()[0].second;
  target_func.realize(res, Halide::Target(), params);

  compare_arrays<Type<DType>>(ref_out_data.data(), res.data(), ref_out_data.size());
}

template <loco::DataType DType>
void fill_data(luci::CircleConst *node, const std::vector<Type<DType>> &data)
{
  assert(node->shape_status() == luci::ShapeStatus::VALID);
  int size = 1;
  for (int i = 0; i < node->rank(); ++i)
    size *= node->dim(i).value();
  node->size<DType>(size);
  assert(data.size() == size);
  for (int i = 0; i < size; ++i)
    node->at<DType>(i) = data[i];
}

template <loco::DataType DType>
void test_const_op(const Shape &shape, DataVector<DType> data)
{
  luci::CircleConst const_node;
  constructBasicNode<DType>(const_node, shape);
  fill_data<DType>(&const_node, data);

  luci::CircleOutput output_node;

  constructBasicNode<DType>(output_node, shape);
  output_node.from(&const_node);

  ASSERT_TRUE(luci_codegen::CodegenKernelBuilder::is_supported(&const_node));

  luci_codegen::SubgraphContext subgraph("", {&const_node});
  subgraph.finish_construction();

  luci_codegen::CodegenKernelBuilder builder(subgraph);
  builder.process();

  Halide::Buffer<Type<DType>> res(reverse_vector(shape));
  Halide::ParamMap params;
  Halide::Func output_func = subgraph.get_outputs()[0].second;

  output_func.realize(res, Halide::Target(), params);

  int output_size = data.size();
  compare_arrays<Type<DType>>(data.data(), res.data(), output_size);
}

TEST(codegen_kernels, constant_scalar_float)
{
  // simple test to check that constant beffers created properly
  std::vector<int> shape{1};
  std::vector<float> data{1.5f};

  test_const_op<loco::DataType::FLOAT32>(shape, data);
}

// TODO add double test

TEST(codegen_kernels, constant_scalar_int32)
{
  // simple test to check that constant beffers created properly
  std::vector<int> shape{1};
  std::vector<int> data{42};

  test_const_op<loco::DataType::S32>(shape, data);
}

TEST(codegen_kernels, constant_scalar_int64)
{
  // simple test to check that constant beffers created properly
  std::vector<int> shape{1};
  std::vector<int64_t > data{10050054321}; // number that do not fit in 32 bit int

  test_const_op<loco::DataType::S64>(shape, data);
}

TEST(codegen_kernels, add_scalar)
{
  std::vector<int> x_shape{1};
  std::vector<int> y_shape{1};
  std::vector<int> out_shape{1};

  std::vector<float> x_data{1.5f};
  std::vector<float> y_data{3.5f};
  std::vector<float> ref_res_data{5.0f};

  test_binary_op<luci::CircleAdd, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, add_scalar_int)
{
  std::vector<int> x_shape{1};
  std::vector<int> y_shape{1};
  std::vector<int> out_shape{1};

  std::vector<int32_t> x_data{1};
  std::vector<int32_t> y_data{3};
  std::vector<int32_t> ref_res_data{4};

  test_binary_op<luci::CircleAdd, loco::DataType::S32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, add_broadcast)
{
  std::vector<int> x_shape{2, 3, 1};
  std::vector<int> y_shape{3, 2};
  std::vector<int> out_shape{2, 3, 2};

  std::vector<float> x_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> y_data{10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
  std::vector<float> ref_res_data{11.f, 21.f, 32.f, 42.f, 53.f, 63.f, 14.f, 24.f, 35.f, 45.f, 56.f, 66.f};

  test_binary_op<luci::CircleAdd, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);

  test_binary_op<luci::CircleAdd, loco::DataType::FLOAT32>(y_shape, x_shape, out_shape, y_data, x_data, ref_res_data);
}

TEST(codegen_kernels, sub_scalar)
{
  std::vector<int> x_shape{1};
  std::vector<int> y_shape{1};
  std::vector<int> out_shape{1};

  std::vector<float> x_data{3.5f};
  std::vector<float> y_data{1.5f};
  std::vector<float> ref_res_data{2.0f};

  test_binary_op<luci::CircleSub, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, sub_broadcast)
{
  std::vector<int> x_shape{2, 3, 1};
  std::vector<int> y_shape{3, 2};
  std::vector<int> out_shape{2, 3, 2};

  std::vector<float> x_data{10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
  std::vector<float> y_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> ref_res_data{9.f, 8.f, 17.f, 16.f, 25.f, 24.f, 39.f, 38.f, 47.f, 46.f, 55.f, 54.f};

  test_binary_op<luci::CircleSub, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, mul_scalar)
{
  std::vector<int> x_shape{1};
  std::vector<int> y_shape{1};
  std::vector<int> out_shape{1};

  std::vector<float> x_data{3.5f};
  std::vector<float> y_data{1.5f};
  std::vector<float> ref_res_data{5.25f};

  test_binary_op<luci::CircleMul, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, mul_broadcast)
{
  std::vector<int> x_shape{2, 3, 1};
  std::vector<int> y_shape{3, 2};
  std::vector<int> out_shape{2, 3, 2};

  std::vector<float> x_data{10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
  std::vector<float> y_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> ref_res_data{10.f, 20.f, 60.f, 80.f, 150.f, 180.f, 40.f, 80.f, 150.f, 200.f, 300.f, 360.f};

  test_binary_op<luci::CircleMul, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, div_scalar)
{
  std::vector<int> x_shape{1};
  std::vector<int> y_shape{1};
  std::vector<int> out_shape{1};

  std::vector<float> x_data{3.5f};
  std::vector<float> y_data{2.f};
  std::vector<float> ref_res_data{1.75f};

  test_binary_op<luci::CircleDiv, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, div_broadcast)
{
  std::vector<int> x_shape{2, 3, 1};
  std::vector<int> y_shape{3, 2};
  std::vector<int> out_shape{2, 3, 2};

  std::vector<float> x_data{10.f, 20.f, 30.f, 40.f, 50.f, 60.f};
  std::vector<float> y_data{1.f, 2.f, 4.f, 8.f, 16.f, 32.f};
  std::vector<float> ref_res_data{10.f, 10.f/2, 20.f/4, 20.f/8, 30.f/16, 30.f/32, 40.f, 40.f/2, 50.f/4, 50.f/8, 60.f/16, 60.f/32};

  test_binary_op<luci::CircleDiv, loco::DataType::FLOAT32>(x_shape, y_shape, out_shape, x_data, y_data, ref_res_data);
}

TEST(codegen_kernels, tanh)
{
  std::vector<int> shape{2, 2};
  std::vector<float> input_data{-1.f, 0.f, 1.5f, 20.f};
  std::vector<float> ref_res_data{-0.7615941559557649f, 0.f, 0.9051482536448664f, 1.f};

  test_unary_op<luci::CircleTanh, loco::DataType::FLOAT32>(shape, input_data, ref_res_data);
}

TEST(codegen_kernels, logistic)
{
  std::vector<int> shape{2, 2};
  std::vector<float> input_data{-1.f, 0.f, 1.5f, 20.f};
  std::vector<float> ref_res_data{0.2689414213699951, 0.5f, 0.8175744761936437, 1.f};

  test_unary_op<luci::CircleLogistic, loco::DataType::FLOAT32>(shape, input_data, ref_res_data);
}

TEST(codegen_kernels, split)
{
  std::vector<int> input_shape{2, 6};
  std::vector<int> out_shape{2, 2};

  std::vector<int32_t> in_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> ref_out1_data{1, 2, 7, 8};
  std::vector<int32_t> ref_out2_data{3, 4, 9, 10};
  std::vector<int32_t> ref_out3_data{5, 6, 11, 12};

  constexpr auto dtype = loco::DataType::S32;
  // construct test graph
  luci::CircleInput input_node;
  constructBasicNode<dtype>(input_node, input_shape);

  luci::CircleConst split_dim;
  constructBasicNode<loco::DataType::S32>(split_dim, {1});
  fill_data<loco::DataType::S32>(&split_dim, {1});

  luci::CircleSplit split;
  split.num_split(3);
  split.split_dim(&split_dim);
  split.input(&input_node);

  luci::CircleSplitOut split_out[3];
  luci::CircleOutput output_node[3];

  for (int i = 0; i < 3; ++i)
  {
    constructBasicNode<dtype>(split_out[i], out_shape);
    split_out[i].index(i);
    split_out[i].input(&split);

    constructBasicNode<dtype>(output_node[i], out_shape);
    output_node[i].from(&split_out[i]);
  }

  ASSERT_TRUE(luci_codegen::CodegenKernelBuilder::is_supported(&split));

  luci_codegen::SubgraphContext subgraph("", {&split, &split_dim, &split_out[0], &split_out[1], &split_out[2]});
  subgraph.finish_construction();

  luci_codegen::CodegenKernelBuilder builder(subgraph);
  builder.process();

  Halide::Buffer<Type<dtype>> input_buffer(in_data.data(), reverse_vector(input_shape));
  std::vector<Halide::Buffer<Type<dtype>>> res;

  for (int i = 0; i < 3; ++i)
    res.emplace_back(reverse_vector(out_shape));

  Halide::ImageParam input_param = subgraph.get_inputs()[0].second;

  Halide::ParamMap params;
  params.set(input_param, input_buffer);

  std::vector<Halide::Func> outputs;
  for (auto output: subgraph.get_outputs())
  {
    outputs.push_back(output.second);
  }
  Halide::Pipeline composite_output(outputs);

  composite_output.realize({res[0], res[1], res[2]}, Halide::Target(), params);

  int output_size = ref_out1_data.size();

  compare_arrays<Type<dtype>>(ref_out1_data.data(), res[0].data(), output_size);
  compare_arrays<Type<dtype>>(ref_out2_data.data(), res[1].data(), output_size);
  compare_arrays<Type<dtype>>(ref_out3_data.data(), res[2].data(), output_size);
}

TEST(codegen_kernels, fc)
{
  std::vector<int> input_shape{1, 3};
  std::vector<int> weights_shape{2, 3};
  std::vector<int> bias_shape{2};
  std::vector<int> output_shape{1, 2};

  std::vector<float> in_data{1.f, 2.f, 3.f};
  std::vector<float> weights_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> ref_out_data{15.f, 34.f};
  std::vector<float> bias_data{1.f, 2.f};

  constexpr auto dtype = loco::DataType::FLOAT32;
  // construct test graph
  luci::CircleInput input_node;
  constructBasicNode<dtype>(input_node, input_shape);

  luci::CircleConst weights_node;
  constructBasicNode<dtype>(weights_node, weights_shape);
  fill_data<loco::DataType::FLOAT32>(&weights_node, weights_data);

  luci::CircleConst bias_node;
  constructBasicNode<dtype>(bias_node, bias_shape);
  fill_data<loco::DataType::FLOAT32>(&bias_node, bias_data);

  luci::CircleFullyConnected fc;
  constructBasicNode<dtype>(fc, output_shape);
  fc.input(&input_node);
  fc.weights(&weights_node);
  fc.bias(&bias_node);
  fc.weights_format(luci::CircleFullyConnected::WeightsFormat::DEFAULT);

  luci::CircleOutput output_node;

  constructBasicNode<dtype>(output_node, output_shape);
  output_node.from(&fc);

  ASSERT_TRUE(luci_codegen::CodegenKernelBuilder::is_supported(&fc));

  luci_codegen::SubgraphContext subgraph("", {&fc, &weights_node, &bias_node});
  subgraph.finish_construction();

  luci_codegen::CodegenKernelBuilder builder(subgraph);
  builder.process();

  Halide::Buffer<Type<dtype>> input_buffer(in_data.data(), reverse_vector(input_shape));
  Halide::Buffer<Type<dtype>> res(reverse_vector(output_shape));

  Halide::ImageParam input_param = subgraph.get_inputs()[0].second;

  Halide::ParamMap params;
  params.set(input_param, input_buffer);

  Halide::Func output_func = subgraph.get_outputs()[0].second;

  output_func.realize(res, Halide::Target(), params);

  int output_size = ref_out_data.size();

  compare_arrays<Type<dtype>>(ref_out_data.data(), res.data(), output_size);
}
