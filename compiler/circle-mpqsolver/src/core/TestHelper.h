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

#ifndef __MPQSOLVER_TEST_HELPER_H__
#define __MPQSOLVER_TEST_HELPER_H__

#include <luci/IR/CircleNodes.h>
#include <luci/IR/Module.h>

class SimpleGraph
{
public:
  SimpleGraph() : _g(loco::make_graph()) {}

public:
  void init()
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

  virtual ~SimpleGraph() = default;
  void transfer_to(luci::Module *module)
  {
    // WARNING: after g is transfered, _graph_inputs, _inputs
    //          and _graph_outputs, _outputs in TestOsGraphlet will be invalid.
    //          arrays are not cleared as this is just helpers to unit tests
    module->add(std::move(_g));
  }

protected:
  virtual loco::Node *insertGraphBody(loco::Node *input) = 0;
  virtual void initInput(loco::Node *input){};

public:
  std::unique_ptr<loco::Graph> _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleOutput *_output = nullptr;
  uint32_t _channel_size = 16;
  uint32_t _width = 4;
  uint32_t _height = 4;
};

#endif //__MPQSOLVER_TEST_HELPER_H__
