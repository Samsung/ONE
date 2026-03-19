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

#include "moco/Import/Nodes/Const.h"
#include "TestHelper.h"

#include <gtest/gtest.h>

using namespace moco::test;

namespace
{

// Test case for "input_tensor.float_val_size() == num_elements"

// clang-format off
const char *const_float_01_pbtxtdata = STRING_CONTENT(
  name: "const/float"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        float_val: 1.1
        float_val: 2.2
        float_val: 3.3
        float_val: 4.4
        float_val: 5.5
        float_val: 6.6
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_float_01)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_float_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - values should match

  tester.inputs({});
  tester.output("const/float");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::FLOAT32>(), 6);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(0), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(1), 2.2f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(2), 3.3f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(3), 4.4f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(4), 5.5f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(5), 6.6f);
}

namespace
{
// Test case for "input_tensor.float_val_size() == 1"

// clang-format off
const char *const_float_02_pbtxtdata = STRING_CONTENT(
  name: "const/float"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        float_val: 1.1
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_float_02)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_float_02_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - values should match

  tester.inputs({});
  tester.output("const/float");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::FLOAT32>(), 6);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(0), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(1), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(2), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(3), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(4), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(5), 1.1f);
}

namespace
{
// Test case for "input_tensor.tensor_content().size() == num_elements * sizeof(float)"
// Generated with tfkit tool: "cat ./test.pbtxt | ./tfkit pack"

// clang-format off
const char *const_float_03_pbtxtdata = STRING_CONTENT(
  name: "const/float"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        tensor_content: "\315\314\214?\315\314\014@33S@\315\314\214@\000\000\260@33\323@"
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_float_03)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_float_03_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - values should match

  tester.inputs({});
  tester.output("const/float");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::FLOAT32>(), 6);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(0), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(1), 2.2f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(2), 3.3f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(3), 4.4f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(4), 5.5f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(5), 6.6f);
}

namespace
{
// Test case for "input_tensor.float_val_size() < num_elements"

// clang-format off
const char *const_float_04_pbtxtdata = STRING_CONTENT(
  name: "const/float"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        float_val: 1.1
        float_val: 2.2
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_float_04)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_float_04_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - values should match

  tester.inputs({});
  tester.output("const/float");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::FLOAT32>(), 6);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(0), 1.1f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(1), 2.2f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(2), 2.2f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(3), 2.2f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(4), 2.2f);
  ASSERT_EQ(test_node->at<loco::DataType::FLOAT32>(5), 2.2f);
}

namespace
{
// Test case for "input_tensor.int_val_size() < num_elements"

// clang-format off
const char *const_int32_04_pbtxtdata = STRING_CONTENT(
  name: "const/int"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        int_val: 1
        int_val: 2
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_int32_04)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_int32_04_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - values should match

  tester.inputs({});
  tester.output("const/int");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::S32>(), 6);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(0), 1);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(1), 2);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(2), 2);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(3), 2);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(4), 2);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(5), 2);
}

namespace
{
// Test case for "scalar"

// clang-format off
const char *const_int32_scalar_pbtxtdata = STRING_CONTENT(
  name: "const/int"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_int32_scalar)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_int32_scalar_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - there should be one element and value should be 3

  tester.inputs({});
  tester.output("const/int");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::S32>(), 1);
  ASSERT_EQ(test_node->at<loco::DataType::S32>(0), 3);
}

namespace
{

// clang-format off
const char *const_int8_01_pbtxtdata = STRING_CONTENT(
  name: "const/int8"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT8
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT8
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 3
          }
        }
        int_val: 0
        int_val: -1
        int_val: 1
        int_val: 2
        int_val: 3
        int_val: 4
      }
    }
  }
);
// clang-format on

} // namespace

TEST(TensorFlowImport, const_int8_01)
{
  TFNodeBuildTester tester;
  moco::ConstGraphBuilder graphbuilder;
  tensorflow::NodeDef nodedef;

  EXPECT_TRUE(plier::tf::parse_nodedef(const_int8_01_pbtxtdata, nodedef));

  // what to test:
  // - there should exist TFConst
  // - values should match

  tester.inputs({});
  tester.output("const/int8");
  tester.run(nodedef, graphbuilder);

  auto test_node = loco::must_cast<moco::TFConst *>(tester.output());
  ASSERT_NE(test_node, nullptr);
  ASSERT_EQ(test_node->size<loco::DataType::S8>(), 6);
  ASSERT_EQ(test_node->at<loco::DataType::S8>(0), 0);
  ASSERT_EQ(test_node->at<loco::DataType::S8>(1), -1);
  ASSERT_EQ(test_node->at<loco::DataType::S8>(2), 1);
  ASSERT_EQ(test_node->at<loco::DataType::S8>(3), 2);
  ASSERT_EQ(test_node->at<loco::DataType::S8>(4), 3);
  ASSERT_EQ(test_node->at<loco::DataType::S8>(5), 4);
}
