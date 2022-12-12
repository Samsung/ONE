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

#include "BisectionSolver.h"
#include "ErrorApproximator.h"
#include "ModuleCloner.h"
#include "Evaluator.h"

#include <cmath>
#include <iostream>

using namespace mpqsolver;

using NodeDepthType = std::map<luci::CircleNode *, float>;

namespace
{

int compute_depth(const luci::Module *module, NodeDepthType &nodes_depth, float &min_depth,
                  float &max_depth)
{
  if (module == nullptr)
    return EXIT_FAILURE;

  if (module->size() != 1)
    return EXIT_FAILURE;

  auto graph = module->graph(0);
  if (!graph)
    return EXIT_FAILURE;

  auto nodes = graph->nodes();
  uint32_t nodes_size = nodes->size();
  std::set<std::string> input_names;
  {
    auto inp_nodes = graph->inputs();
    for (uint32_t i = 0; i < inp_nodes->size(); ++i)
    {
      auto inp_node = inp_nodes->at(i);
      auto inp_name = inp_node->name();
      input_names.insert(inp_name);
    }
  }

  // initializing
  std::vector<luci::CircleNode *> to_process;
  std::map<std::string, float> named_depth;
  {
    auto inputs = loco::input_nodes(graph);
    for (auto &node : inputs)
    {
      auto cnode = loco::must_cast<luci::CircleNode *>(node);
      to_process.emplace_back(cnode);
      nodes_depth[cnode] = 0.f;
      named_depth[cnode->name()] = 0.f;
    }
  }

  // enumerating
  while (!to_process.empty())
  {
    auto cur_node = to_process.back();
    to_process.pop_back();
    auto iter = nodes_depth.find(cur_node);
    if (iter == nodes_depth.end())
    {
      return EXIT_FAILURE; // unexpected
    }
    float cur_depth = iter->second + 1;
    // processing children
    auto children = loco::succs(cur_node);
    for (auto &child : children)
    {
      auto cichild = loco::must_cast<luci::CircleNode *>(child);
      auto node_depth = nodes_depth.find(cichild);
      if (node_depth == nodes_depth.end() || node_depth->second < cur_depth)
      {
        // initialize depth
        nodes_depth[cichild] = cur_depth;
        to_process.push_back(cichild);
        named_depth[cichild->name()] = cur_depth;
      }
    }
  }

  auto minmax = std::minmax_element(
    nodes_depth.begin(), nodes_depth.end(),
    [=](const std::pair<luci::CircleNode *, float> &el1,
        const std::pair<luci::CircleNode *, float> &el2) { return el1.second < el2.second; });

  min_depth = minmax.first->second;
  max_depth = minmax.second->second;

  return EXIT_SUCCESS;
}

bool error_at_input_is_larger_than_at_output(const NodeDepthType &nodes_depth, float cut_depth)
{
  float error_at_input = 0;
  float error_at_output = 0;
  for (auto &iter : nodes_depth)
  {
    float cur_error = ErrorApproximator::approximate(iter.first);
    if (iter.second < cut_depth)
    {
      error_at_input += cur_error;
    }
    else
    {
      error_at_output += cur_error;
    }
  }

  if (error_at_input > error_at_output)
  {
    std::cerr << "Q16 will be set at input due to ";
  }
  else
  {
    std::cerr << "Q8 will be set at input due to ";
  }
  std::cerr << error_at_input << " error at input vs ";
  std::cerr << error_at_output << " error at output." << std::endl;

  return error_at_input > error_at_output;
}

class BisectionOptionsImpl final : public BisectionSolver::Options
{
public:
  virtual void enable(Q16AtInput) final;
  virtual bool query(Q16AtInput) final;

private:
  Q16AtInput _q16AtInput = Q16AtInput::Auto;
};

void BisectionOptionsImpl::enable(Q16AtInput q16AtInput) { _q16AtInput = q16AtInput; }
bool BisectionOptionsImpl::query(Q16AtInput q16AtInput) { return _q16AtInput == q16AtInput; }

} // namespace

BisectionSolver::BisectionSolver(const std::string &input_data_path, float qerror_ratio)
  : _input_data_path(input_data_path), _qerror_ratio(qerror_ratio), _qerror(0.f)
{
}

BisectionSolver::Options *BisectionSolver::options(void)
{
  if (_options == nullptr)
  {
    _options = std::make_unique<BisectionOptionsImpl>();
  }

  return _options.get();
}

std::unique_ptr<luci::Module> BisectionSolver::run(const luci::Module *in_module)
{
  float min_depth = 0.f;
  float max_depth = 0.f;
  NodeDepthType nodes_depth;
  if (compute_depth(in_module, nodes_depth, min_depth, max_depth) != EXIT_SUCCESS)
  {
    std::cerr << "Invalid graph for bisectioning" << std::endl;
    return nullptr;
  }

  DatasetEvaluator evaluator(in_module, _input_data_path);
  // const auto &ref_output = compute_outputs(module.get(), _input_data_path);
  LayerParams layer_params;
  float int16_qerror = evaluator.evaluate("int16", layer_params);
  std::cerr << "int16_quantization_error " << int16_qerror << std::endl;

  float int8_qerror = evaluator.evaluate("uint8", layer_params);
  std::cerr << "int8_quantization_error " << int8_qerror << std::endl;

  _qerror = int16_qerror + _qerror_ratio * std::fabs(int8_qerror - int16_qerror);
  std::cerr << "target quantization error " << _qerror << std::endl;

  int last_depth = -1;
  float best_depth = -1;
  LayerParams best_params;
  auto graph = in_module->graph(0);
  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));
  // input and output nodes are not valid for quantization, so let's remove them
  for (auto node : loco::input_nodes(graph))
  {
    active_nodes.erase(node);
  }
  for (auto node : loco::output_nodes(graph))
  {
    active_nodes.erase(node);
  }

  // let's decide whether nodes at input are more suspectible to be quantized into Q16, than at
  // output
  bool int16_at_input = true;
  if (auto option = options())
  {
    if (option->query(Options::Q16AtInput::Auto))
    {
      int16_at_input =
        error_at_input_is_larger_than_at_output(nodes_depth, 0.5f * (max_depth + min_depth));
    }
    else if (option->query(Options::Q16AtInput::True))
    {
      int16_at_input = true;
    }
    else if (option->query(Options::Q16AtInput::False))
    {
      int16_at_input = false;
    }
  }

  while (true)
  {
    float cut_depth = 0.5f * (min_depth + max_depth);

    if (last_depth == int(cut_depth))
    {
      break;
    }
    last_depth = int(cut_depth);

    auto nodes = graph->nodes();
    LayerParams layer_params;
    for (auto &node : active_nodes)
    {
      auto cur_node = loco::must_cast<luci::CircleNode *>(node);
      auto iter = nodes_depth.find(cur_node);
      if (iter == nodes_depth.end())
      {
        continue;
      }

      float depth = iter->second;
      if ((depth <= cut_depth && int16_at_input) || (depth >= cut_depth && !int16_at_input))
      {
        auto layer_param = std::make_shared<LayerParam>();
        {
          layer_param->name = cur_node->name();
          layer_param->dtype = "int16";
          layer_param->granularity = "channel";
        }

        layer_params.emplace_back(layer_param);
      }
    }

    float cur_accuracy = evaluator.evaluate("uint8", layer_params);
    std::cerr << cut_depth << " : " << cur_accuracy << std::endl;

    if (cur_accuracy < _qerror)
    {
      int16_at_input ? (max_depth = cut_depth) : (min_depth = cut_depth);
      best_params = layer_params;
      best_depth = cut_depth;
    }
    else
    {
      int16_at_input ? (min_depth = cut_depth) : (max_depth = cut_depth);
    }
  }

  if (best_params.empty())
  {
    std::cerr << "Failed to optimal configuration" << std::endl;
    return nullptr;
  }

  std::cerr << "Found optimal configuration at " << best_depth << " depth." << std::endl;
  return evaluator.quantize("uint8", best_params);
}
