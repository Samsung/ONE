/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "LayerInfoMap.h"

#include <luci/IR/CircleNode.h>
#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

class SoftmaxTestGraph : public luci::test::TestIOGraph
{
public:
  void init(void)
  {
    TestIOGraph::init({32}, {32});
    _softmax = g()->nodes()->create<luci::CircleSoftmax>();
    {
      _softmax->logits(input());
      _softmax->beta(0.1);
      _softmax->name("test");
    }
    output()->from(_softmax);
  }

private:
  luci::CircleSoftmax *_softmax = nullptr;
};

} // namespace

TEST(LayerInfoMapTest, simple_test)
{
  SoftmaxTestGraph g;
  g.init();

  luci::LayerInfo info;
  {
    info.name = "test";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  auto map = luci::layer_info_map(g.g(), v);

  EXPECT_EQ("test", map["test"]->name);
  EXPECT_EQ(loco::DataType::U8, map["test"]->dtype);
  EXPECT_EQ(luci::QuantizationGranularity::ChannelWise, map["test"]->granularity);
}

TEST(LayerInfoMapTest, duplicate_name_NEG)
{
  SoftmaxTestGraph g;
  g.init();
  g.input()->name("test");

  luci::LayerInfo info;
  {
    info.name = "test";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  EXPECT_ANY_THROW(luci::layer_info_map(g.g(), v));
}

TEST(LayerInfoMapTest, no_name_NEG)
{
  SoftmaxTestGraph g;
  g.init();

  luci::LayerInfo info;
  {
    info.name = "noname";
    info.dtype = loco::DataType::U8;
    info.granularity = luci::QuantizationGranularity::ChannelWise;
  }
  std::vector<luci::LayerInfo> v;
  v.emplace_back(info);
  EXPECT_ANY_THROW(luci::layer_info_map(g.g(), v));
}
