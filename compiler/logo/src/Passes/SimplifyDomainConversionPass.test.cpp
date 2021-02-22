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

#include <logo/SimplifyDomainConversionPass.h>

#include "TestHelper.h"

#include <loco.h>

#include <memory>

#include <gtest/gtest.h>

TEST(SimplifyDomainConversionPassTest, name)
{
  logo::SimplifyDomainConversionPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(SimplifyDomainConversionPassTest, run_NEG)
{
  loco::Graph g;
  logo::SimplifyDomainConversionPass pass;

  ASSERT_FALSE(pass.run(&g));
}

namespace
{

// code borrowed from GraphBlock.h/cpp in exo-tflite
enum class FilterLayout
{
  OHWI, // a.k.a., NHWC, Tensorflow Lite uses this layout
  HWIO, // Tensorflow format
};

template <FilterLayout T> loco::Permutation<loco::Domain::Filter> perm();

template <> loco::Permutation<loco::Domain::Filter> perm<FilterLayout::OHWI>()
{
  // Make NHWC permutation for encoder and decoder
  loco::Permutation<loco::Domain::Filter> OHWI; // a.k.a., NHWC

  OHWI.axis(loco::FilterAxis::Count) = 0;
  OHWI.axis(loco::FilterAxis::Height) = 1;
  OHWI.axis(loco::FilterAxis::Width) = 2;
  OHWI.axis(loco::FilterAxis::Depth) = 3;

  return OHWI;
}

template <> loco::Permutation<loco::Domain::Filter> perm<FilterLayout::HWIO>()
{
  // Make NHWC permutation for encoder and decoder
  loco::Permutation<loco::Domain::Filter> HWIO;

  HWIO.axis(loco::FilterAxis::Height) = 0;
  HWIO.axis(loco::FilterAxis::Width) = 1;
  HWIO.axis(loco::FilterAxis::Depth) = 2;
  HWIO.axis(loco::FilterAxis::Count) = 3;

  return HWIO;
}

template <FilterLayout T> loco::FilterDecode *make_filter_decode(loco::Node *input_for_decode)
{
  loco::Graph *g = input_for_decode->graph();

  auto decoder = std::make_unique<loco::PermutingDecoder<loco::Domain::Filter>>();

  decoder->perm(perm<T>());

  auto dec = g->nodes()->create<loco::FilterDecode>();
  dec->input(input_for_decode);
  dec->decoder(std::move(decoder));

  return dec;
}

template <FilterLayout T> loco::FilterEncode *make_filter_encode(loco::Node *input_for_encode)
{
  loco::Graph *g = input_for_encode->graph();

  auto encoder = std::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>();

  encoder->perm(perm<T>());

  auto enc = g->nodes()->create<loco::FilterEncode>();
  enc->input(input_for_encode);
  enc->encoder(std::move(encoder));

  return enc;
}

/*
  test case:
      ConstGen (2x3x4x5) ---- FeatureEncode ---- FeatureDecode --- Push
          0                         H                 O              0
          1                         W                 H              1
          2                         I(depth)          W              2
          3                         O(count)          I              3

      axis 0 ---------------------> H --------------> H -----------> 1
      axis 1 ---------------------> W --------------> W -----------> 2
      axis 2 ---------------------> I --------------> I -----------> 3
      axis 3 ---------------------> O --------------> O -----------> 0

      so perm vector of Tranpose = [3, 0, 1, 2]
*/
void create_net_FilterEncode_FilterDecode_different_perms(loco::Graph *graph)
{
  assert(graph);

  auto const_node = graph->nodes()->create<loco::ConstGen>();
  {
    const_node->dtype(loco::DataType::FLOAT32);
    const_node->rank(4);
    int count = 1;
    for (int i = 0; i < 4; ++i)
    {
      const_node->dim(i) = i + 2;
      count *= i + 2;
    }
    const_node->size<loco::DataType::FLOAT32>(count);
    for (uint32_t i = 0; i < count; i++)
      const_node->at<loco::DataType::FLOAT32>(i) = 3.14f; // any number
  }

  auto encoder = make_filter_encode<FilterLayout::HWIO>(const_node);
  auto decoder = make_filter_decode<FilterLayout::OHWI>(encoder);

  auto push_node = graph->nodes()->create<loco::Push>();
  {
    push_node->from(decoder);
  }

  auto graph_output = graph->outputs()->create();
  {
    graph_output->name("output");
    graph_output->dtype(loco::DataType::FLOAT32);
    loco::link(graph_output, push_node);
  }
}

/*
  test case:
      ConstGen (2x3x4x5) ---- FeatureEncode ---- FeatureDecode --- Push
          0                         H                 H              0
          1                         W                 W              1
          2                         I(depth)          I              2
          3                         O(count)          O              3

      axis 0 ---------------------> H --------------> H -----------> 0
      axis 1 ---------------------> W --------------> W -----------> 1
      axis 2 ---------------------> I --------------> I -----------> 2
      axis 3 ---------------------> O --------------> O -----------> 3

      so perm vector of Tranpose = [0, 1, 2, 3] and transposes should be eliminated
*/
void create_net_FilterEncode_FilterDecode_equal_perms(loco::Graph *graph)
{
  assert(graph);

  auto const_node = graph->nodes()->create<loco::ConstGen>();
  {
    const_node->dtype(loco::DataType::FLOAT32);
    const_node->rank(4);
    int count = 1;
    for (int i = 0; i < 4; ++i)
    {
      const_node->dim(i) = i + 2;
      count *= i + 2;
    }
    const_node->size<loco::DataType::FLOAT32>(count);
    for (uint32_t i = 0; i < count; i++)
      const_node->at<loco::DataType::FLOAT32>(i) = 3.14f; // any number
  }

  auto encoder = make_filter_encode<FilterLayout::HWIO>(const_node);
  auto decoder = make_filter_decode<FilterLayout::HWIO>(encoder);

  auto push_node = graph->nodes()->create<loco::Push>();
  {
    push_node->from(decoder);
  }

  auto graph_output = graph->outputs()->create();
  {
    graph_output->name("output");
    graph_output->dtype(loco::DataType::FLOAT32);
    loco::link(graph_output, push_node);
  }
}

} // namespace

TEST(SimplifyDomainConversionPass, FilterEncode_FilterDecode_different_perms)
{
  auto graph = loco::make_graph();
  create_net_FilterEncode_FilterDecode_different_perms(graph.get());

  logo::SimplifyDomainConversionPass pass;
  while (pass.run(graph.get()) == true)
    ;

  auto tr = logo::test::find_first_node_by_type<loco::TensorTranspose>(graph.get());
  {
    ASSERT_EQ(tr->perm()->size(), 4);
    ASSERT_EQ(tr->perm()->axis(0), 3);
    ASSERT_EQ(tr->perm()->axis(1), 0);
    ASSERT_EQ(tr->perm()->axis(2), 1);
    ASSERT_EQ(tr->perm()->axis(3), 2);
  }

  auto const_gen = dynamic_cast<loco::ConstGen *>(tr->input());
  ASSERT_NE(const_gen, nullptr);
}

TEST(SimplifyDomainConversionPass, FilterEncode_FilterDecode_equal_perms)
{
  auto graph = loco::make_graph();
  create_net_FilterEncode_FilterDecode_equal_perms(graph.get());

  logo::SimplifyDomainConversionPass pass;
  while (pass.run(graph.get()) == true)
    ;

  ASSERT_EQ(loco::output_nodes(graph.get()).size(), 1);
  loco::Node *output_node = loco::output_nodes(graph.get())[0];

  auto forward = loco::must_cast<loco::Forward *>(output_node->arg(0));
  ASSERT_NE(forward, nullptr);
  auto const_gen = dynamic_cast<loco::ConstGen *>(forward->arg(0));
  ASSERT_NE(const_gen, nullptr);
}
