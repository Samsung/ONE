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

#include "locoex/COpCall.h"
#include "locoex/COpAttrTypes.h"

#include <loco/IR/Graph.h>
#include <loco/IR/Nodes.h>

#include <memory>

#include <gtest/gtest.h>

TEST(CallTest, Test_01)
{
  using namespace locoex;

  // attr name
  std::string int_attr = "my_int";
  std::string float_attr = "my_float";

  int int_val = 100;
  float float_val = 3.14;

  // building loco test graph
  auto g = loco::make_graph();

  // generating input
  auto inp = g->nodes()->create<loco::Pull>();
  {
    inp->dtype(loco::DataType::FLOAT32);
    inp->shape({1, 2});
  }

  // generating custom op
  auto custom = g->nodes()->create<COpCall>(2U);
  {
    custom->input(0, inp);
    custom->input(1, inp);

    custom->attr(int_attr, std::make_unique<COpAttrInt>(int_val));
    custom->attr(float_attr, std::make_unique<COpAttrFloat>(float_val));
  }

  // access custom op input
  loco::Node *input0 = custom->input(0);
  loco::Node *input1 = custom->input(1);

  ASSERT_EQ(custom->arity(), 2);
  ASSERT_EQ(dynamic_cast<loco::Pull *>(input0), inp);
  ASSERT_EQ(dynamic_cast<loco::Pull *>(input1), inp);

  // access custom op attrs
  auto names = custom->attr_names();

  bool int_cheched = false, float_cheched = false;

  for (const auto &name : names)
  {
    if (auto int_attr = custom->attr<COpAttrType::Int>(name))
    {
      ASSERT_EQ(int_attr->val(), int_val);
      int_cheched = true;
    }
    else if (auto float_attr = custom->attr<COpAttrType::Float>(name))
    {
      ASSERT_FLOAT_EQ(float_attr->val(), float_val);
      float_cheched = true;
    }
    else
    {
      FAIL();
    }
  }

  ASSERT_TRUE(int_cheched && float_cheched);
}
