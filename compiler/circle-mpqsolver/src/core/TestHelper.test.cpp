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

} // namespace io_utils
} // namespace test
} // namespace mpqsolver
