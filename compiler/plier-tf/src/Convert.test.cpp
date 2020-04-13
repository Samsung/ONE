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

#include <plier/tf/Convert.h>

#include <tensorflow/core/framework/graph.pb.h>

#include <gtest/gtest.h>

#include <string>

namespace
{

void prepare_test_node(tensorflow::NodeDef &node)
{
  node.set_op("Placeholder");
  node.set_name("node");

  tensorflow::AttrValue dtype_attr;
  dtype_attr.set_type(tensorflow::DT_FLOAT);
  (*node.mutable_attr())["dtype_1"] = dtype_attr;

  auto *shape = (*node.mutable_attr())["shape_1"].mutable_shape();
  shape->add_dim()->set_size(1);
  shape->add_dim()->set_size(2);
  shape->add_dim()->set_size(4);
  shape->add_dim()->set_size(8);

  auto *list = (*node.mutable_attr())["list_1"].mutable_list();
  list->add_i(1);
  list->add_i(20);
  list->add_i(1LL << 40);
  list->add_i(-(1LL << 40));
}

} // namespace

TEST(plier_Convert, attr)
{
  tensorflow::NodeDef node;
  prepare_test_node(node);

  ASSERT_TRUE(plier::tf::has_attr(node, "dtype_1"));
  ASSERT_FALSE(plier::tf::has_attr(node, "other"));
}

TEST(plier_Convert, attr_datatype)
{
  tensorflow::NodeDef node;
  prepare_test_node(node);

  ASSERT_EQ(plier::tf::get_datatype_attr(node, "dtype_1"), tensorflow::DT_FLOAT);
}

TEST(plier_Convert, attr_shape)
{
  tensorflow::NodeDef node;
  prepare_test_node(node);

  const auto &shape = plier::tf::get_shape_attr(node, "shape_1");
  ASSERT_EQ(shape.dim_size(), 4);
  ASSERT_EQ(shape.dim(0).size(), 1);
  ASSERT_EQ(shape.dim(1).size(), 2);
  ASSERT_EQ(shape.dim(2).size(), 4);
  ASSERT_EQ(shape.dim(3).size(), 8);
}

TEST(plier_Convert, to_loco_datatype)
{
  ASSERT_EQ(plier::tf::as_loco_datatype(tensorflow::DT_FLOAT), loco::DataType::FLOAT32);
}

TEST(plier_Convert, attr_ilist)
{
  tensorflow::NodeDef node;
  prepare_test_node(node);

  const auto &p_list = plier::tf::get_list_attr(node, "list_1");
  auto i_list = plier::tf::as_int64_list(p_list);
  ASSERT_EQ(i_list.size(), 4);
  ASSERT_EQ(i_list.at(0), 1);
  ASSERT_EQ(i_list.at(1), 20);
  ASSERT_EQ(i_list.at(2), 1LL << 40);
  ASSERT_EQ(i_list.at(3), -(1LL << 40));
}

TEST(plier_Convert, to_data_layout)
{
  ASSERT_EQ(plier::tf::as_data_layout("NHWC"), plier::tf::DataLayout::NHWC);
  ASSERT_EQ(plier::tf::as_data_layout("NCHW"), plier::tf::DataLayout::NCHW);
}

TEST(plier_Convert, copy_shape_thrown_on_unknown_dim)
{
  tensorflow::TensorShapeProto tf_shape;
  nncc::core::ADT::tensor::Shape angkor_shape;

  tf_shape.add_dim()->set_size(-1);

  ASSERT_ANY_THROW(plier::tf::copy_shape(tf_shape, angkor_shape));
}
