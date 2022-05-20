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

#include "MetricPrinter.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using Tensor = circle_eval_diff::Tensor;

namespace
{

// TODO Reduce duplicate codes in ResolveCustomOpMatMulPass.cpp
template <typename T>
luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<T> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape.at(i);
    size *= shape.at(i);
  }
  node->shape_status(luci::ShapeStatus::VALID);

#define INIT_VALUES(DT)                          \
  {                                              \
    node->size<DT>(size);                        \
    for (uint32_t i = 0; i < values.size(); ++i) \
      node->at<DT>(i) = values[i];               \
  }

  switch (dtype)
  {
    case loco::DataType::U8:
      INIT_VALUES(loco::DataType::U8);
      break;
    case loco::DataType::S16:
      INIT_VALUES(loco::DataType::S16);
      break;
    case loco::DataType::S32:
      INIT_VALUES(loco::DataType::S32);
      break;
    case loco::DataType::FLOAT32:
      INIT_VALUES(loco::DataType::FLOAT32)
      break;
    default:
      INTERNAL_EXN("create_const_node called with unsupported type");
      break;
  }
  return node;
}

/**
 *  Simple graph which adds constant (addition) to the input
 *
 *  [Input] [Const] (addition)
 *      \   /
 *      [Add]
 *
 */
class AddGraphlet
{
public:
  AddGraphlet() = default;

  void init(loco::Graph *g, float addition)
  {
    std::vector<float> addition_val;
    for (uint32_t i = 0; i < 16; i++)
      addition_val.push_back(addition);
    _add_c = create_const_node(g, loco::DataType::FLOAT32, {1, 16}, addition_val);

    _add = g->nodes()->create<luci::CircleAdd>();
    _add->y(_add_c);
    _add->fusedActivationFunction(luci::FusedActFunc::NONE);
    _add->dtype(loco::DataType::FLOAT32);
    _add->shape({1, 16});
    _add->name("add");
  }

protected:
  luci::CircleAdd *_add = nullptr;
  luci::CircleConst *_add_c = nullptr;
};

class AddOneGraph : public luci::test::TestIOGraph, public AddGraphlet
{
public:
  AddOneGraph() = default;

  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4}, {1, 16});
    AddGraphlet::init(g(), 1.0);

    _add->x(input());

    output()->from(_add);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

class AddTwoGraph : public luci::test::TestIOGraph, public AddGraphlet
{
public:
  AddTwoGraph() = default;

  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4}, {1, 16});
    AddGraphlet::init(g(), 2.0);

    _add->x(input());

    output()->from(_add);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

// Return number of elements of the node.
uint32_t numElements(const luci::CircleNode *node)
{
  uint32_t num_elem = 1;
  for (uint32_t i = 0; i < node->rank(); ++i)
    num_elem *= node->dim(i).value();
  return num_elem;
}

// Return Tensor which has the same dtype and shape with node.
// Buffer does not have any data yet.
std::shared_ptr<Tensor> create_empty_tensor(const luci::CircleNode *node)
{
  auto tensor = std::make_shared<Tensor>();
  {
    tensor->dtype(node->dtype());
    tensor->rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); i++)
      tensor->dim(i) = node->dim(i);
    tensor->size<loco::DataType::FLOAT32>(numElements(node));
  }

  return tensor;
}

std::shared_ptr<Tensor> output_tensor_with_value(const luci::Module *module, float value)
{
  auto outputs = loco::output_nodes(module->graph());
  assert(outputs.size() == 1);
  auto output = *outputs.begin();
  auto output_cnode = loco::must_cast<luci::CircleNode *>(output);
  auto tensor = create_empty_tensor(output_cnode);
  auto tensor_size = tensor->size<loco::DataType::FLOAT32>();
  for (uint32_t i = 0; i < tensor_size; i++)
  {
    tensor->at<loco::DataType::FLOAT32>(i) = value;
  }
  return tensor;
}

std::shared_ptr<Tensor> output_tensor_with_value(const luci::Module *module,
                                                 std::vector<float> &value)
{
  auto outputs = loco::output_nodes(module->graph());
  assert(outputs.size() == 1);
  auto output = *outputs.begin();
  auto output_cnode = loco::must_cast<luci::CircleNode *>(output);
  auto tensor = create_empty_tensor(output_cnode);
  auto tensor_size = tensor->size<loco::DataType::FLOAT32>();
  assert(tensor_size == value.size());
  for (uint32_t i = 0; i < tensor_size; i++)
  {
    tensor->at<loco::DataType::FLOAT32>(i) = value[i];
  }
  return tensor;
}

} // namespace

namespace circle_eval_diff
{

TEST(CircleEvalMetricPrinterTest, MAE_simple)
{
  luci::Module first;
  AddOneGraph first_g;
  first_g.init();

  first.add(std::move(first_g.graph()));

  luci::Module second;
  AddTwoGraph second_g;
  second_g.init();

  second.add(std::move(second_g.graph()));

  MAEPrinter mae;

  mae.init(&first, &second);

  // This test does not actually evaluate the modules, but create
  // fake results.
  std::vector<std::shared_ptr<Tensor>> first_result;
  {
    auto output = output_tensor_with_value(&first, 1.0);
    first_result.emplace_back(output);
  }

  std::vector<std::shared_ptr<Tensor>> second_result;
  {
    auto output = output_tensor_with_value(&second, 2.0);
    second_result.emplace_back(output);
  }

  mae.accumulate(first_result, second_result);

  std::stringstream ss;
  mae.dump(ss);
  std::string result = ss.str();

  EXPECT_NE(std::string::npos, result.find("MAE for output_0 is 1"));
}

TEST(CircleEvalMetricPrinterTest, MAE_init_with_null_NEG)
{
  MAEPrinter mae;

  EXPECT_ANY_THROW(mae.init(nullptr, nullptr));
}

TEST(CircleEvalMetricPrinterTest, MAPE_simple)
{
  luci::Module first;
  AddOneGraph first_g;
  first_g.init();

  first.add(std::move(first_g.graph()));

  luci::Module second;
  AddTwoGraph second_g;
  second_g.init();

  second.add(std::move(second_g.graph()));

  MAPEPrinter mape;

  mape.init(&first, &second);

  // This test does not actually evaluate the modules, but create
  // fake results.
  std::vector<std::shared_ptr<Tensor>> first_result;
  {
    auto output = output_tensor_with_value(&first, 2.0);
    first_result.emplace_back(output);
  }

  std::vector<std::shared_ptr<Tensor>> second_result;
  {
    auto output = output_tensor_with_value(&second, 1.0);
    second_result.emplace_back(output);
  }

  mape.accumulate(first_result, second_result);

  std::stringstream ss;
  mape.dump(ss);
  std::string result = ss.str();

  EXPECT_NE(std::string::npos, result.find("MAPE for output_0 is 50%"));
}

TEST(CircleEvalMetricPrinterTest, MAPE_init_with_null_NEG)
{
  MAPEPrinter mape;

  EXPECT_ANY_THROW(mape.init(nullptr, nullptr));
}

TEST(CircleEvalMetricPrinterTest, MPEIR_simple)
{
  luci::Module first;
  AddOneGraph first_g;
  first_g.init();

  first.add(std::move(first_g.graph()));

  luci::Module second;
  AddTwoGraph second_g;
  second_g.init();

  second.add(std::move(second_g.graph()));

  MPEIRPrinter mpeir;

  mpeir.init(&first, &second);

  // This test does not actually evaluate the modules, but create
  // fake results.
  std::vector<std::shared_ptr<Tensor>> first_result;
  {
    std::vector<float> val;
    val.resize(16);
    for (uint32_t i = 0; i < 16; i++)
      val[i] = i;

    auto output = output_tensor_with_value(&first, val);
    first_result.emplace_back(output);
  }

  std::vector<std::shared_ptr<Tensor>> second_result;
  {
    auto output = output_tensor_with_value(&second, 0.0);
    second_result.emplace_back(output);
  }

  mpeir.accumulate(first_result, second_result);

  std::stringstream ss;
  mpeir.dump(ss);
  std::string result = ss.str();

  EXPECT_NE(std::string::npos, result.find("MPEIR for output_0 is 1"));
}

TEST(CircleEvalMetricPrinterTest, MPEIR_init_with_null_NEG)
{
  MPEIRPrinter mpeir;

  EXPECT_ANY_THROW(mpeir.init(nullptr, nullptr));
}

TEST(CircleEvalMetricPrinterTest, TopK_simple)
{
  luci::Module first;
  AddOneGraph first_g;
  first_g.init();

  first.add(std::move(first_g.graph()));

  luci::Module second;
  AddTwoGraph second_g;
  second_g.init();

  second.add(std::move(second_g.graph()));

  TopKMatchPrinter top5(5);

  top5.init(&first, &second);

  // This test does not actually evaluate the modules, but create
  // fake results.
  std::vector<std::shared_ptr<Tensor>> first_result;
  {
    std::vector<float> val;
    val.resize(16);
    for (uint32_t i = 0; i < 16; i++)
      val[i] = i;

    auto output = output_tensor_with_value(&first, val);
    first_result.emplace_back(output);
  }

  std::vector<std::shared_ptr<Tensor>> second_result;
  {
    std::vector<float> val;
    val.resize(16);
    for (uint32_t i = 0; i < 16; i++)
      val[i] = i * 2;
    auto output = output_tensor_with_value(&second, val);
    second_result.emplace_back(output);
  }

  top5.accumulate(first_result, second_result);

  std::stringstream ss;
  top5.dump(ss);
  std::string result = ss.str();

  EXPECT_NE(std::string::npos, result.find("Mean Top-5 match ratio for output_0 is 1"));
}

TEST(CircleEvalMetricPrinterTest, TopK_tie)
{
  luci::Module first;
  AddOneGraph first_g;
  first_g.init();

  first.add(std::move(first_g.graph()));

  luci::Module second;
  AddTwoGraph second_g;
  second_g.init();

  second.add(std::move(second_g.graph()));

  TopKMatchPrinter top5(5);

  top5.init(&first, &second);

  // This test does not actually evaluate the modules, but create
  // fake results.
  std::vector<std::shared_ptr<Tensor>> first_result;
  {
    std::vector<float> val;
    val.resize(16);
    for (uint32_t i = 0; i < 16; i++)
      val[i] = i;

    auto output = output_tensor_with_value(&first, val);
    first_result.emplace_back(output);
  }

  std::vector<std::shared_ptr<Tensor>> second_result;
  {
    std::vector<float> val{12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 15, 16};

    auto output = output_tensor_with_value(&second, val);
    second_result.emplace_back(output);
  }

  top5.accumulate(first_result, second_result);

  std::stringstream ss;
  top5.dump(ss);
  std::string result = ss.str();

  EXPECT_NE(std::string::npos, result.find("Mean Top-5 match ratio for output_0 is 0.8"));
}

TEST(CircleEvalMetricPrinterTest, TopK_num_elem_less_than_k_NEG)
{
  luci::Module first;
  AddOneGraph first_g;
  first_g.init();

  first.add(std::move(first_g.graph()));

  luci::Module second;
  AddTwoGraph second_g;
  second_g.init();

  second.add(std::move(second_g.graph()));

  TopKMatchPrinter top100(100);

  top100.init(&first, &second);

  // This test does not actually evaluate the modules, but create
  // fake results.
  std::vector<std::shared_ptr<Tensor>> first_result;
  {
    auto output = output_tensor_with_value(&first, 0);
    first_result.emplace_back(output);
  }

  std::vector<std::shared_ptr<Tensor>> second_result;
  {
    auto output = output_tensor_with_value(&second, 0);
    second_result.emplace_back(output);
  }

  top100.accumulate(first_result, second_result);

  std::stringstream ss;
  top100.dump(ss);
  std::string result = ss.str();

  EXPECT_EQ(std::string::npos, result.find("Mean Top-100 match ratio"));
}

TEST(CircleEvalMetricPrinterTest, TopK_init_with_null_NEG)
{
  TopKMatchPrinter topk(5);

  EXPECT_ANY_THROW(topk.init(nullptr, nullptr));
}

} // namespace circle_eval_diff
