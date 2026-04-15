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

#include "moco/Importer.h"
#include "moco/GraphHelper.h"

#include <moco/IR/Nodes/TFIdentity.h>

#include "TestHelper.h"
#include <loco.h>
#include <plier/tf/TestHelper.h>

#include <gtest/gtest.h>

using namespace moco::test;

TEST(TensorFlowImport, Dummy) { moco::Importer import; }

namespace
{

// clang-format off
const char *basic_pbtxtdata = STRING_CONTENT(
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 2
        }
        dim {
          size: 1
        }
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "output/identity"
  op: "Identity"
  input: "Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
);
// clang-format on

} // namespace

TEST(TensorFlowImport, load_model_withio_tf)
{
  moco::ModelSignature signature;

  signature.add_input(moco::TensorName("Placeholder", 0));
  signature.add_output(moco::TensorName("output/identity", 0));

  tensorflow::GraphDef graph_def;
  EXPECT_TRUE(plier::tf::parse_graphdef(basic_pbtxtdata, graph_def));

  moco::Importer importer;

  std::unique_ptr<loco::Graph> graph = importer.import(signature, graph_def);

  // what to test:
  // - import reads Placeholder
  // - import reads Identity
  // - attribute values should match

  auto tfidentity = find_first_node_bytype<moco::TFIdentity>(graph.get());
  ASSERT_NE(tfidentity, nullptr);
  ASSERT_NE(tfidentity->input(), nullptr);
}

namespace
{

// clang-format off
const char *query_pbtxtdata = STRING_CONTENT(
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 2
        }
        dim {
          size: 1
        }
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "Foo/w_min"
  op: "Const"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape { }
        float_val: -1.0
      }
    }
  }
}
node {
  name: "output/identity"
  op: "Identity"
  input: "Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "Foo/w_max"
  op: "Const"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape { }
        float_val: -1.0
      }
    }
  }
}
);
// clang-format on

} // namespace

TEST(TensorFlowImport, find_node_by_name)
{
  moco::ModelSignature signature;

  signature.add_input(moco::TensorName("Placeholder", 0));
  signature.add_output(moco::TensorName("output/identity", 0));

  tensorflow::GraphDef graph_def;
  EXPECT_TRUE(plier::tf::parse_graphdef(query_pbtxtdata, graph_def));

  moco::Importer importer;

  std::unique_ptr<loco::Graph> graph = importer.import(signature, graph_def);

  // what to test:
  // - get name of first Identity node
  // - find node by name `Foo/w_min`
  // - find node by name `Foo/w_max`

  auto tfidentity = find_first_node_bytype<moco::TFIdentity>(graph.get());
  ASSERT_NE(tfidentity, nullptr);
  ASSERT_NE(tfidentity->input(), nullptr);
  ASSERT_STREQ(tfidentity->name().c_str(), "output/identity");

  auto query_node = moco::find_node_byname<moco::TFConst>(graph.get(), "Foo/w_min");
  ASSERT_NE(query_node, nullptr);
  ASSERT_STREQ(query_node->name().c_str(), "Foo/w_min");

  auto query_node2 = moco::find_node_byname<moco::TFConst>(graph.get(), "Foo/w_max");
  ASSERT_NE(query_node2, nullptr);
  ASSERT_STREQ(query_node2->name().c_str(), "Foo/w_max");
}
