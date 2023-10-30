/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TestHelper.h"

#include <fstream>
#include <H5Cpp.h>

namespace mpqsolver
{
namespace test
{
namespace models
{

void SimpleGraph::init()
{
  _input = _g->nodes()->create<luci::CircleInput>();
  _output = _g->nodes()->create<luci::CircleOutput>();
  _input->name("input");
  _output->name("output");

  auto graph_input = _g->inputs()->create();
  _input->index(graph_input->index());
  auto graph_output = _g->outputs()->create();
  _output->index(graph_output->index());

  graph_input->dtype(loco::DataType::FLOAT32);
  _input->dtype(loco::DataType::FLOAT32);
  _output->dtype(loco::DataType::FLOAT32);
  graph_output->dtype(loco::DataType::FLOAT32);

  graph_input->shape({1, _height, _width, _channel_size});
  _input->shape({1, _height, _width, _channel_size});
  _output->shape({1, _height, _width, _channel_size});
  graph_output->shape({1, _height, _width, _channel_size});

  auto graph_body = insertGraphBody(_input);
  _output->from(graph_body);

  initInput(_input);
}

void SimpleGraph::transfer_to(luci::Module *module)
{
  // WARNING: after g is transfered, _graph_inputs, _inputs
  //          and _graph_outputs, _outputs in TestOsGraphlet will be invalid.
  //          arrays are not cleared as this is just helpers to unit tests
  module->add(std::move(_g));
}

void AddGraph::initInput(loco::Node *input)
{
  auto ci_input = loco::must_cast<luci::CircleNode *>(input);
  initMinMax(ci_input);
}

void AddGraph::initMinMax(luci::CircleNode *node)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  qparam->min.assign(1, _a_min);
  qparam->max.assign(1, _a_max);
  node->quantparam(std::move(qparam));
}

loco::Node *AddGraph::insertGraphBody(loco::Node *input)
{
  _add = _g->nodes()->create<luci::CircleAdd>();
  _beta = _g->nodes()->create<luci::CircleConst>();

  _add->dtype(loco::DataType::FLOAT32);
  _beta->dtype(loco::DataType::FLOAT32);

  _add->shape({1, _height, _width, _channel_size});
  _beta->shape({1, _height, _width, _channel_size});

  _beta->size<loco::DataType::FLOAT32>(_channel_size * _width * _height);
  _add->x(input);
  _add->y(_beta);
  _add->fusedActivationFunction(luci::FusedActFunc::NONE);

  _add->name("add");
  _beta->name("beta");
  initMinMax(_add);

  return _add;
}

void SoftmaxGraphlet::initMinMax(luci::CircleNode *node, float min, float max)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  qparam->min.assign(1, min);
  qparam->max.assign(1, max);
  node->quantparam(std::move(qparam));
}

void SoftmaxGraphlet::init(loco::Graph *g)
{
  _ifm = nullptr;

  _ifm = g->nodes()->create<luci::CircleAbs>();
  _max = g->nodes()->create<luci::CircleReduceMax>();
  _sub = g->nodes()->create<luci::CircleSub>();
  _exp = g->nodes()->create<luci::CircleExp>();
  _sum = g->nodes()->create<luci::CircleSum>();
  _div = g->nodes()->create<luci::CircleDiv>();
  _softmax_indices = g->nodes()->create<luci::CircleConst>();

  _ifm->name("ifm");
  _max->name("maximum_of_ifm");
  _sub->name("sub");
  _exp->name("exp");
  _sum->name("sum");
  _div->name("div");
  _softmax_indices->name("reduction_indices");

  initMinMax(_ifm, 0, 1);
  initMinMax(_max, 0, 1);
  initMinMax(_sub, 0, 1);
  initMinMax(_exp, 0, 1);
  initMinMax(_sum, 0, 1);
  initMinMax(_div, 0, 1);

  _softmax_indices->dtype(loco::DataType::S32);
  _softmax_indices->size<loco::DataType::S32>(1);
  _softmax_indices->shape({1});
  _softmax_indices->at<loco::DataType::S32>(0) = -1;
  _softmax_indices->shape_status(luci::ShapeStatus::VALID);

  _max->keep_dims(true);
  _sum->keep_dims(true);
}

void SoftmaxTestGraph::init(void)
{
  TestIOGraph::init({1, 12, 11, 15}, {1, 12, 11, 15});
  SoftmaxGraphlet::init(g());

  _ifm->x(input());
  _max->input(_ifm);
  _max->reduction_indices(_softmax_indices);

  _sub->x(_ifm);
  _sub->y(_max);
  _sub->fusedActivationFunction(luci::FusedActFunc::NONE);
  _exp->x(_sub);
  _sum->input(_exp);
  _sum->reduction_indices(_softmax_indices);
  _div->x(_exp);
  _div->y(_sum);
  _div->fusedActivationFunction(luci::FusedActFunc::NONE);

  output()->from(_div);

  initMinMax(input(), 0, 1);
  initMinMax(output(), 0, 1);
}

} // namespace models

namespace io_utils
{

void makeTemporaryFile(char *name_template)
{
  int fd = mkstemp(name_template);
  if (fd == -1)
  {
    throw std::runtime_error{"mkstemp failed"};
  }
}

void writeDataToFile(const std::string &path, const std::string &data)
{
  std::ofstream file;
  file.open(path);
  file << data;
  file.close();
}

std::string makeTemporaryFolder(char *name_template)
{
  auto const res = mkdtemp(name_template);
  if (res == nullptr)
  {
    throw std::runtime_error{"mkdtemp failed"};
  }
  return res;
}

bool isFileExists(const std::string &path)
{
  std::ifstream f(path);
  return f.good();
}

} // namespace io_utils
} // namespace test
} // namespace mpqsolver
