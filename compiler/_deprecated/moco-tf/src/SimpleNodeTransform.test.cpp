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

#include "SimpleNodeTransform.h"

#include <set>

#include <gtest/gtest.h>

TEST(SimpleNodeTransformTests, run)
{
  class Transform final : public moco::tf::SimpleNodeTransform<loco::Push>
  {
  public:
    Transform(std::multiset<loco::Node *> *out) : _out{out}
    {
      // DO NOTHING
    }

  public:
    bool transform(loco::Push *node) const final
    {
      _out->insert(node);
      return false;
    }

  private:
    std::multiset<loco::Node *> *_out;
  };

  auto g = loco::make_graph();
  auto output_0 = g->outputs()->create();
  auto push = g->nodes()->create<loco::Push>();
  loco::link(output_0, push);

  std::multiset<loco::Node *> nodes;
  Transform transform{&nodes};

  transform.run(g.get());

  ASSERT_EQ(nodes.size(), 1);
  ASSERT_EQ(nodes.count(push), 1);
}
