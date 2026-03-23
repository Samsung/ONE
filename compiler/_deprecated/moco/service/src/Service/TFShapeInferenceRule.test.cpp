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

#include "moco/Service/TFShapeInferenceRule.h"

#include "TestHelper.h"

#include "moco/IR/TFNodes.h"

#include <loco.h>
#include <loco/Service/ShapeInference.h>

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

moco::TFAvgPool *avgpool_network_simple1331(loco::Graph *graph)
{
  auto avgpool_node = graph->nodes()->create<moco::TFAvgPool>();

  avgpool_node->data_layout("NHWC");
  avgpool_node->ksize({1, 3, 3, 1});
  avgpool_node->strides({1, 1, 1, 1});

  // Dummy const node as ifm, just to fake TFShapeInferenceRule for TFAvgPool.
  auto const_node = graph->nodes()->create<moco::TFConst>();
  {
    const_node->rank(4);
    const_node->dim(0).set(1);
    const_node->dim(1).set(3);
    const_node->dim(2).set(3);
    const_node->dim(3).set(1);
  }
  avgpool_node->value(const_node);

  setup_output_node(graph, avgpool_node);

  return avgpool_node;
}

} // namespace

TEST(TFShapeInferenceRule, avgpool_same)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto avgpool_node = avgpool_network_simple1331(&graph);
  avgpool_node->padding("SAME");

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(avgpool_node);
  auto tshape = nodeshape.as<loco::TensorShape>();
  ASSERT_EQ(tshape.rank(), 4);
  ASSERT_EQ(tshape.dim(0).value(), 1);
  ASSERT_EQ(tshape.dim(1).value(), 3);
  ASSERT_EQ(tshape.dim(2).value(), 3);
  ASSERT_EQ(tshape.dim(3).value(), 1);
}

TEST(TFShapeInferenceRule, avgpool_valid)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto avgpool_node = avgpool_network_simple1331(&graph);
  avgpool_node->padding("VALID");

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(avgpool_node);
  auto tshape = nodeshape.as<loco::TensorShape>();
  ASSERT_EQ(tshape.rank(), 4);
  ASSERT_EQ(tshape.dim(0).value(), 1);
  ASSERT_EQ(tshape.dim(1).value(), 1);
  ASSERT_EQ(tshape.dim(2).value(), 1);
  ASSERT_EQ(tshape.dim(3).value(), 1);
}

namespace
{

void conv2d_test(const std::array<uint32_t, 4> ifm_shape, const std::array<uint32_t, 4> ker_shape,
                 const std::array<uint32_t, 2> stride_h_w, std::string padding,
                 const std::array<uint32_t, 4> expected_shape)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto conv2d_node = graph.nodes()->create<moco::TFConv2D>();
  conv2d_node->data_layout("NHWC");
  conv2d_node->strides({1, stride_h_w[0], stride_h_w[1], 1});
  conv2d_node->padding(padding);

  auto ifm_node = graph.nodes()->create<moco::TFConst>();
  {
    ifm_node->rank(4);
    ifm_node->dim(0).set(ifm_shape[0]);
    ifm_node->dim(1).set(ifm_shape[1]);
    ifm_node->dim(2).set(ifm_shape[2]);
    ifm_node->dim(3).set(ifm_shape[3]);
  }

  auto ker_node = graph.nodes()->create<moco::TFConst>();
  {
    ker_node->rank(4);
    ker_node->dim(0).set(ker_shape[0]);
    ker_node->dim(1).set(ker_shape[1]);
    ker_node->dim(2).set(ker_shape[2]);
    ker_node->dim(3).set(ker_shape[3]);
  }

  conv2d_node->input(ifm_node);
  conv2d_node->filter(ker_node);

  setup_output_node(&graph, conv2d_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(conv2d_node);
  auto tshape = nodeshape.as<loco::TensorShape>();
  ASSERT_EQ(tshape.rank(), 4);
  ASSERT_EQ(tshape.dim(0).value(), expected_shape[0]);
  ASSERT_EQ(tshape.dim(1).value(), expected_shape[1]);
  ASSERT_EQ(tshape.dim(2).value(), expected_shape[2]);
  ASSERT_EQ(tshape.dim(3).value(), expected_shape[3]);
}

} // namespace

/*
  Testing "InceptionV3/InceptionV3/Conv2d_1a_3x3/Conv2D" Conv2D node in Inception_v3:
  The result shape of this test is generated with the code below:

    ifm = tf.constant(value=1.1, shape=[1, 299, 299, 3])
    ker = tf.constant(value=1.1, shape=[3, 3, 3, 32])

    out = tf.nn.conv2d(ifm, ker, strides = [1, 2, 2, 1], padding= 'VALID')

    with tf.Session() as sess:
        res = sess.run(out)
        print(res.shape)
 */
TEST(TFShapeInferenceRule, conv2d_VALID)
{
  conv2d_test({1, 299, 299, 3},   // ifm
              {3, 3, 3, 32},      // ker
              {2, 2},             // strides
              "VALID",            // padding
              {1, 149, 149, 32}); // expected shape after FixShape
}

/*
  Testing "InceptionV3/InceptionV3/Conv2d_2b_3x3/Conv2D" Conv2D node in Inception_v3:
  The result shape of this test is generated with the code below:

    ifm = tf.constant(value=1.1, shape=[1, 147, 147, 32])
    ker = tf.constant(value=1.1, shape=[3, 3, 32, 64])

    out = tf.nn.conv2d(ifm, ker, strides = [1, 1, 1, 1], padding= 'SAME')

    with tf.Session() as sess:
        res = sess.run(out)
        print(res.shape)
 */
TEST(TFShapeInferenceRule, conv2d_SAME)
{
  conv2d_test({1, 147, 147, 32},  // ifm
              {3, 3, 32, 64},     // ker
              {1, 1},             // strides
              "SAME",             // padding
              {1, 147, 147, 64}); // expected shape after FixShape
}

/*
  Testing Pack
*/
namespace
{

moco::TFConst *const_scalar(loco::Graph *graph, int32_t val)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(0);
  const_node->size<loco::DataType::S32>(1);
  const_node->at<loco::DataType::S32>(0) = val;

  return const_node;
}

moco::TFConst *const_vector(loco::Graph *graph, int32_t dim)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(1);
  const_node->dim(0).set(dim);

  const_node->size<loco::DataType::S32>(dim);
  for (int32_t i = 0; i < dim; ++i)
    const_node->at<loco::DataType::S32>(i) = i;

  return const_node;
}

moco::TFConst *const_vector_init(loco::Graph *graph, std::vector<int32_t> values)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();
  auto dim = values.size();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(1);
  const_node->dim(0).set(dim);

  const_node->size<loco::DataType::S32>(dim);
  for (int32_t i = 0; i < dim; ++i)
    const_node->at<loco::DataType::S32>(i) = values[i];

  return const_node;
}

moco::TFConst *const_matrix(loco::Graph *graph, int32_t dimh, int32_t dimw)
{
  auto const_node = graph->nodes()->create<moco::TFConst>();

  const_node->dtype(loco::DataType::S32);
  const_node->rank(2);
  const_node->dim(0).set(dimh);
  const_node->dim(1).set(dimw);

  auto elements = dimh * dimw;
  const_node->size<loco::DataType::S32>(elements);
  for (int32_t i = 0; i < elements; ++i)
    const_node->at<loco::DataType::S32>(i) = i;

  return const_node;
}

} // namespace

TEST(TFShapeInferenceRule, pack_scalar_2)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(2);
  pack_node->axis(0);
  {
    auto const_node_0 = const_scalar(&graph, 1);
    pack_node->values(0, const_node_0);
    auto const_node_1 = const_scalar(&graph, 1);
    pack_node->values(1, const_node_1);
  }
  setup_output_node(&graph, pack_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(pack_node);
  auto tshape = nodeshape.as<loco::TensorShape>();
  ASSERT_EQ(tshape.rank(), 1);
  ASSERT_EQ(tshape.dim(0).value(), 2);
}

TEST(TFShapeInferenceRule, pack_vector3_2)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(2);
  pack_node->axis(0);
  {
    auto const_node_0 = const_vector(&graph, 3);
    pack_node->values(0, const_node_0);
    auto const_node_1 = const_vector(&graph, 3);
    pack_node->values(1, const_node_1);
  }
  setup_output_node(&graph, pack_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(pack_node);
  auto tshape = nodeshape.as<loco::TensorShape>();

  ASSERT_EQ(tshape.rank(), 2);
  ASSERT_EQ(tshape.dim(0).value(), 2);
  ASSERT_EQ(tshape.dim(1).value(), 3);
}

TEST(TFShapeInferenceRule, pack_vector3_2_axis_1)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(2);
  pack_node->axis(1);
  {
    auto const_node_0 = const_vector(&graph, 3);
    pack_node->values(0, const_node_0);
    auto const_node_1 = const_vector(&graph, 3);
    pack_node->values(1, const_node_1);
  }
  setup_output_node(&graph, pack_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(pack_node);
  auto tshape = nodeshape.as<loco::TensorShape>();

  ASSERT_EQ(tshape.rank(), 2);
  ASSERT_EQ(tshape.dim(0).value(), 3);
  ASSERT_EQ(tshape.dim(1).value(), 2);
}

TEST(TFShapeInferenceRule, pack_vector3_2_axis_m2)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(2);
  pack_node->axis(-2);
  {
    auto const_node_0 = const_vector(&graph, 3);
    pack_node->values(0, const_node_0);
    auto const_node_1 = const_vector(&graph, 3);
    pack_node->values(1, const_node_1);
  }
  setup_output_node(&graph, pack_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(pack_node);
  auto tshape = nodeshape.as<loco::TensorShape>();

  ASSERT_EQ(tshape.rank(), 2);
  ASSERT_EQ(tshape.dim(0).value(), 2);
  ASSERT_EQ(tshape.dim(1).value(), 3);
}

TEST(TFShapeInferenceRule, pack_vector3_2_axis_m3)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(2);
  pack_node->axis(-3);
  {
    auto const_node_0 = const_vector(&graph, 3);
    pack_node->values(0, const_node_0);
    auto const_node_1 = const_vector(&graph, 3);
    pack_node->values(1, const_node_1);
  }
  setup_output_node(&graph, pack_node);

  // -3 is out of range and should throw
  EXPECT_ANY_THROW(loco::apply(&shape_infer).to(&graph));
}

TEST(TFShapeInferenceRule, pack_matrix3x4_2_axis_1)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto pack_node = graph.nodes()->create<moco::TFPack>(2);
  pack_node->axis(1);
  {
    auto const_node_0 = const_matrix(&graph, 3, 4);
    pack_node->values(0, const_node_0);
    auto const_node_1 = const_matrix(&graph, 3, 4);
    pack_node->values(1, const_node_1);
  }
  setup_output_node(&graph, pack_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(pack_node);
  auto tshape = nodeshape.as<loco::TensorShape>();

  ASSERT_EQ(tshape.rank(), 3);
  ASSERT_EQ(tshape.dim(0).value(), 3);
  ASSERT_EQ(tshape.dim(1).value(), 2);
  ASSERT_EQ(tshape.dim(2).value(), 4);
}

TEST(TFShapeInferenceRule, stridedslice_matrix5x5_shrink)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto sslice_node = graph.nodes()->create<moco::TFStridedSlice>();
  {
    auto const_input = const_matrix(&graph, 5, 5);
    sslice_node->input(const_input);

    auto const_begin = const_vector_init(&graph, {1, 1});
    sslice_node->begin(const_begin);
    auto const_end = const_vector_init(&graph, {2, 4});
    sslice_node->end(const_end);
    auto const_strides = const_vector_init(&graph, {1, 1});
    sslice_node->strides(const_strides);

    sslice_node->shrink_axis_mask(1);
  }
  setup_output_node(&graph, sslice_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(sslice_node);
  auto tshape = nodeshape.as<loco::TensorShape>();

  ASSERT_EQ(tshape.rank(), 1);
  ASSERT_EQ(tshape.dim(0).value(), 3);
}

TEST(TFShapeInferenceRule, stridedslice_4_shrink)
{
  moco::TFShapeInferenceRule shape_infer;
  loco::Graph graph;

  auto sslice_node = graph.nodes()->create<moco::TFStridedSlice>();
  {
    auto const_input = const_vector(&graph, 4);
    sslice_node->input(const_input);

    auto const_begin = const_vector_init(&graph, {0});
    sslice_node->begin(const_begin);
    auto const_end = const_vector_init(&graph, {1});
    sslice_node->end(const_end);
    auto const_strides = const_vector_init(&graph, {1});
    sslice_node->strides(const_strides);

    sslice_node->shrink_axis_mask(1);
  }
  setup_output_node(&graph, sslice_node);

  bool cont = true;
  while (cont)
  {
    cont = loco::apply(&shape_infer).to(&graph);
  };

  auto nodeshape = loco::shape_get(sslice_node);
  auto tshape = nodeshape.as<loco::TensorShape>();

  ASSERT_EQ(tshape.rank(), 0);
}
