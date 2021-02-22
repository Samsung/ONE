/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "TFLExporterImpl.h"

#include "schema_generated.h"

#include "TestGraph.h"
#include "GraphBlock.h"
#include "Knob.h"

#include <loco/IR/PermutingCodec.h>

#include <memory>

#include <gtest/gtest.h>

namespace
{

class TFLExporterImplTests : public ::testing::Test
{
public:
  TFLExporterImplTests() { _graph = loco::make_graph(); }

public:
  virtual ~TFLExporterImplTests() = default;

protected:
  loco::Graph *graph(void) { return _graph.get(); }

  template <typename NodeT> NodeT *make_node(void);

private:
  std::unique_ptr<loco::Graph> _graph;
};

template <typename NodeT> NodeT *TFLExporterImplTests::make_node(void)
{
  return graph()->nodes()->create<NodeT>();
}

template <> loco::FeatureEncode *TFLExporterImplTests::make_node(void)
{
  loco::FeatureEncode *encode_layer = graph()->nodes()->create<loco::FeatureEncode>();

  auto encoder = std::make_unique<loco::PermutingEncoder<loco::Domain::Feature>>();
  (*encoder->perm())[loco::FeatureAxis::Count] = 0;
  (*encoder->perm())[loco::FeatureAxis::Depth] = 1;
  (*encoder->perm())[loco::FeatureAxis::Height] = 2;
  (*encoder->perm())[loco::FeatureAxis::Width] = 3;
  encode_layer->encoder(std::move(encoder));

  return encode_layer;
}

template <> loco::FeatureDecode *TFLExporterImplTests::make_node(void)
{
  loco::FeatureDecode *decode_layer = graph()->nodes()->create<loco::FeatureDecode>();

  auto decoder = std::make_unique<loco::PermutingDecoder<loco::Domain::Feature>>();
  (*decoder->perm())[loco::FeatureAxis::Count] = 0;
  (*decoder->perm())[loco::FeatureAxis::Depth] = 1;
  (*decoder->perm())[loco::FeatureAxis::Height] = 2;
  (*decoder->perm())[loco::FeatureAxis::Width] = 3;
  decode_layer->decoder(std::move(decoder));

  return decode_layer;
}

} // namespace

// TODO TFLAdd

// TODO TFLAveragePool2D

TEST_F(TFLExporterImplTests, Concatenate)
{
  auto pull1 = make_node<loco::Pull>();
  {
    pull1->dtype(loco::DataType::FLOAT32);
    pull1->shape({1, 2, 3, 4});
  }
  auto pull2 = make_node<loco::Pull>();
  {
    pull2->dtype(loco::DataType::FLOAT32);
    pull2->shape({1, 2, 3, 4});
  }
  auto concat = make_node<loco::TensorConcat>();
  {
    concat->lhs(pull1);
    concat->rhs(pull2);
  }
  auto push = make_node<loco::Push>();
  {
    push->from(concat);
  }

  auto input1 = graph()->inputs()->create();
  {
    input1->name("input1");
    loco::link(input1, pull1);
  }
  auto input2 = graph()->inputs()->create();
  {
    input2->name("input2");
    loco::link(input2, pull2);
  }
  auto output = graph()->outputs()->create();
  {
    output->name("output");
    loco::link(output, push);
  }

  exo::TFLExporter::Impl exporter{graph()};

  // TODO Add more checks
  SUCCEED();
}

// TODO TFLConv2D

// TODO TFLDepthwiseConv2D

// TODO TFLDiv

// TODO TFLMaxPool2D

// TODO TFLMul

TEST_F(TFLExporterImplTests, Relu6)
{
  auto pull = make_node<loco::Pull>();
  {
    pull->dtype(loco::DataType::FLOAT32);
    pull->shape({1, 8, 8, 3});
  }
  auto relu6 = make_node<loco::ReLU6>();
  {
    relu6->input(pull);
  }
  auto push = make_node<loco::Push>();
  {
    push->from(relu6);
  }

  auto input = graph()->inputs()->create();
  {
    input->name("input");
    loco::link(input, pull);
  }
  auto output = graph()->outputs()->create();
  {
    output->name("output");
    loco::link(output, push);
  }

  exo::TFLExporter::Impl exporter{graph()};

  // TODO Add more checks
  SUCCEED();
}

// TODO TFLRelu6

// TODO TFLReshape

// TODO TFLSoftmax

// TODO TFLSqrt

// TODO TFLSub

// TODO TFLTanh

TEST(TFLExporterImplTest, Transpose_simple)
{
  exo::test::ExampleGraph<exo::test::ExampleGraphType::Transpose> g;

  // pull attribute
  {
    g.pull->dtype(loco::DataType::FLOAT32);
    g.pull->shape({1, 2, 2, 3});
  }

  // transpose attribute
  {
    g.transpose->perm()->size(4);
    g.transpose->perm()->axis(0) = 1;
    g.transpose->perm()->axis(1) = 2;
    g.transpose->perm()->axis(2) = 3;
    g.transpose->perm()->axis(3) = 0;
  }

  exo::TFLExporter::Impl exporter{g.graph()};
  {
    auto model = tflite::GetModel(exporter.getBufferPointer());
    auto operators = model->subgraphs()->Get(0)->operators();

    assert(operators->Length() == 1);

    int n = 0; // op index of Transpose in tflite file

    auto opcode_index = operators->Get(n)->opcode_index();

    ASSERT_EQ(model->operator_codes()->Get(opcode_index)->builtin_code(),
              tflite::BuiltinOperator_TRANSPOSE);

    auto perm = operators->Get(n)->inputs()->Get(1);

    auto perm_tensor = model->subgraphs()->Get(0)->tensors()->Get(perm);
    ASSERT_EQ(tflite::TensorType::TensorType_INT32, perm_tensor->type());
    ASSERT_EQ(1, perm_tensor->shape()->size());
    ASSERT_EQ(4, perm_tensor->shape()->Get(0));

    auto bufs = (model->buffers());
    auto *perm_buf =
      reinterpret_cast<const int32_t *>(bufs->Get(perm_tensor->buffer())->data()->data());

    ASSERT_EQ(1, perm_buf[0]);
    ASSERT_EQ(2, perm_buf[1]);
    ASSERT_EQ(3, perm_buf[2]);
    ASSERT_EQ(0, perm_buf[3]);
  }
}

/*
  test case:
        Pull ----- FeatureEncode ---- FeatureDecode --- Push
          0 -----------> H ---------+      O              0
          1              W          +----> H -----------> 1
          2              I(depth)          W              2
          3              O(coutn)          I              3

      axis 0 ----------> H --------------> H -----------> 1
      axis 1 ----------> W --------------> W -----------> 2
      axis 2 ----------> I --------------> I -----------> 3
      axis 3 ----------> O --------------> O -----------> 0

      So, perm vector of Tranpose = [3, 0, 1, 2].
      Please refer to loco::TensorTranspose about the definition of perm vector.
*/
TEST(TFLExporterImplTest, Transpose_from_FilterEncode_FilterDecode)
{
  exo::test::ExampleGraph<exo::test::ExampleGraphType::FilterEncode_FilterDecode> g;

  // pull attribute
  {
    g.pull->dtype(loco::DataType::FLOAT32);
    g.pull->shape({1, 2, 3, 4}); // whatever value of rank 4
  }

  exo::TFLExporter::Impl exporter{g.graph()};
  {
    auto model = tflite::GetModel(exporter.getBufferPointer());
    auto operators = model->subgraphs()->Get(0)->operators();

    assert(operators->Length() == 1);

    int n = 0; // op index of Transpose in tflite file

    auto opcode_index = operators->Get(n)->opcode_index();

    ASSERT_EQ(model->operator_codes()->Get(opcode_index)->builtin_code(),
              tflite::BuiltinOperator_TRANSPOSE);

    auto perm = operators->Get(n)->inputs()->Get(1);

    auto perm_tensor = model->subgraphs()->Get(0)->tensors()->Get(perm);
    ASSERT_EQ(tflite::TensorType::TensorType_INT32, perm_tensor->type());
    ASSERT_EQ(1, perm_tensor->shape()->size());
    ASSERT_EQ(4, perm_tensor->shape()->Get(0));

    auto bufs = (model->buffers());
    auto *perm_buf =
      reinterpret_cast<const int32_t *>(bufs->Get(perm_tensor->buffer())->data()->data());
    ASSERT_EQ(3, perm_buf[0]);
    ASSERT_EQ(0, perm_buf[1]);
    ASSERT_EQ(1, perm_buf[2]);
    ASSERT_EQ(2, perm_buf[3]);
  }
}

/**
 * What happens when there is a mismatch between generation and execution order!?
 */
TEST_F(TFLExporterImplTests, Regression_0000)
{
  // This test was written without considering fusion.
  // For this reason, this check is needed.
  // TODO Rewrite this test
  if (exo::get<exo::Knob::UseFuseReluPass>())
    return;

  // Execution Order: MaxPool2D -> ReLU
  // Generation Order: ReLU -> MaxPool2D
  auto pull = make_node<loco::Pull>();
  {
    pull->dtype(loco::DataType::FLOAT32);
    pull->shape({1, 8, 8, 3});
  }
  auto relu = make_node<loco::ReLU>();
  auto encode = exo::make_feature_encode<exo::FeatureLayout::NHWC>(pull);
  auto maxpool = make_node<loco::MaxPool2D>();
  auto decode = exo::make_feature_decode<exo::FeatureLayout::NHWC>(relu);
  auto push = make_node<loco::Push>();

  ASSERT_EQ(1, maxpool->window()->vertical());
  ASSERT_EQ(1, maxpool->window()->horizontal());

  maxpool->ifm(encode);
  relu->input(maxpool);
  push->from(decode);

  auto input = graph()->inputs()->create();
  {
    input->name("input");
    loco::link(input, pull);
  }
  auto output = graph()->outputs()->create();
  {
    output->name("output");
    loco::link(output, push);
  }

  exo::TFLExporter::Impl exporter{graph()};
  {
    int64_t maxpool_execution_index = -1;
    int64_t relu_exeuction_index = -1;

    auto model = tflite::GetModel(exporter.getBufferPointer());
    auto operators = model->subgraphs()->Get(0)->operators();

    for (uint32_t n = 0; n < operators->Length(); ++n)
    {
      auto opcode_index = operators->Get(n)->opcode_index();

      switch (model->operator_codes()->Get(opcode_index)->builtin_code())
      {
        case tflite::BuiltinOperator_RELU:
          ASSERT_EQ(-1, relu_exeuction_index);
          relu_exeuction_index = static_cast<int64_t>(n);
          break;
        case tflite::BuiltinOperator_MAX_POOL_2D:
          ASSERT_EQ(-1, maxpool_execution_index);
          maxpool_execution_index = static_cast<int64_t>(n);
          break;
        default:
          break;
      }
    }

    ASSERT_NE(maxpool_execution_index, -1);
    ASSERT_NE(relu_exeuction_index, -1);
    // maxpool SHOULD precede ReLU
    ASSERT_LT(maxpool_execution_index, relu_exeuction_index);
  }
}

/**
 * @brief  Test exporter buffer generation
 */
TEST_F(TFLExporterImplTests, Regression_0001)
{
  auto cgen = make_node<loco::ConstGen>();
  cgen->rank(1);
  cgen->dim(0) = 2;
  cgen->dtype(loco::DataType::FLOAT32);
  cgen->size<loco::DataType::FLOAT32>(2);
  cgen->at<loco::DataType::FLOAT32>(0) = 3.3f;
  cgen->at<loco::DataType::FLOAT32>(1) = 1.1f;

  auto push = make_node<loco::Push>();
  push->from(cgen);

  auto output = graph()->outputs()->create();
  {
    output->name("output");
    loco::link(output, push);
  }

  exo::TFLExporter::Impl exporter{graph()};
  {
    auto model = tflite::GetModel(exporter.getBufferPointer());
    auto buffers = model->buffers();

    // 0'th empty buffer + ConstGen data + ConstGen node output
    ASSERT_EQ(3, buffers->Length());

    // 0'th should be empty buffer
    auto buffer_0 = (*buffers)[0];
    auto array_0 = buffer_0->data();
    ASSERT_EQ(nullptr, array_0);

    // 1'st should be ConstGen data which is two float
    auto buffer_1 = (*buffers)[1];
    auto array_1 = buffer_1->data();
    size_t size_1 = array_1->size();
    ASSERT_EQ(2 * sizeof(float), size_1);
  }
}
