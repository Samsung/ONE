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

#include "COpCall.h"

#include "TestHelper.h"

#include "Canonicalizer.h"

#include <moco/Importer.h>

#include <locoex/COpCall.h>
#include <locoex/COpAttrTypes.h>

#include <loco.h>
#include <plier/tf/TestHelper.h>

#include <gtest/gtest.h>

#include <memory>

using namespace moco::tf::test;

namespace
{
// clang-format off
const char *customop_01_pbtxtdata = STRING_CONTENT(
node {
  name: "input1"
  op: "Placeholder"
  attr {
    key: "dtype" value { type: DT_FLOAT } }
  attr {
    key: "shape"
    value { shape { dim { size: 1 } dim { size: 2 } } }
  }
}
node {
  name: "input2"
  op: "Const"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape { dim { size: 1 } dim { size: 2 } }
        float_val: 1.1 float_val: 2.2
      }
    }
  }
}
node {
  name: "my/customOp/000"
  op: "new_custom_op"
  input: "input1"
  input: "input2"
  attr { key: "my_float"  value { f: 0.001 } }
  attr { key: "my_int"    value { i: 111 } }
}
);

// clang-format on
} // namespace

TEST(Call_Test, Call_01)
{
  moco::ModelSignature signature;
  {
    signature.add_input(moco::TensorName("input1", 0));
    signature.add_output(moco::TensorName("my/customOp/000", 0));
    signature.add_customop("new_custom_op");
    signature.dtype("my/customOp/000", loco::DataType::FLOAT32);
    signature.shape("my/customOp/000", {1, 2});
  }

  tensorflow::GraphDef graph_def;
  EXPECT_TRUE(plier::tf::parse_graphdef(customop_01_pbtxtdata, graph_def));

  // import
  moco::GraphBuilderRegistry registry{&moco::GraphBuilderRegistry::get()};
  registry.add("new_custom_op", std::make_unique<moco::tf::COpCallGraphBuilder>(&signature));

  moco::Importer importer(&registry);
  std::unique_ptr<loco::Graph> graph = importer.import(signature, graph_def);

  // what to test:
  // - there should exist COpCall
  // - two input nodes should exist and not be nullptr
  // - attributes should match

  auto *customop = moco::tf::test::find_first_node_bytype<locoex::COpCall>(graph.get());
  ASSERT_NE(customop, nullptr);

  ASSERT_EQ(customop->arity(), 2);

  loco::Node *input_0 = customop->arg(0);
  loco::Node *input_1 = customop->arg(1);
  ASSERT_NE(input_0, nullptr);
  ASSERT_NE(input_1, nullptr);

  auto f_attr = customop->attr<locoex::COpAttrType::Float>("my_float");
  ASSERT_FLOAT_EQ(f_attr->val(), 0.001);
  ASSERT_TRUE(f_attr->type() == locoex::COpAttrType::Float);

  auto i_attr = customop->attr<locoex::COpAttrType::Int>("my_int");
  ASSERT_FLOAT_EQ(i_attr->val(), 111);
  ASSERT_TRUE(i_attr->type() == locoex::COpAttrType::Int);
}
