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

#include <H5Cpp.h>

#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/ImporterEx.h>
#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleQuantParam.h>
#include <luci/Profile/CircleNodeID.h>
#include <luci/Service/Validate.h>

#include <arser/arser.h>
#include <vconone/vconone.h>

#include "h5/Reader.h"

#include <cassert>
#include <cmath> // for std::floor
#include <iostream>
#include <string>

namespace
{

void print_version(void)
{
  std::cout << "minmax-embedder version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

/* NOTE: getNthPercentile is copied from compiler/record-minmax/include/RecordFunction.h */
/**
 * @brief  getNthPercentile calculates the n-th percentile of input vector (0.0 <= n <= 100.0)
 *         linear interpolation is used when the desired percentile lies between two data points
 */
float getNthPercentile(std::vector<float> &vector, float percentile)
{
  if (percentile < 0 || percentile > 100)
    throw std::runtime_error("Percentile must be ranged from 0 to 100");

  if (vector.empty())
    throw std::runtime_error("Percentile must take a non-empty vector as an argument");

  if (vector.size() == 1)
    return vector[0];

  std::vector<float> copy;
  copy.assign(vector.begin(), vector.end());
  std::sort(copy.begin(), copy.end());

  if (percentile == 0.0)
    return copy.front();

  if (percentile == 100.0)
    return copy.back();

  int index = static_cast<int>(std::floor((copy.size() - 1) * percentile / 100.0));

  float percent_i = static_cast<float>(index) / static_cast<float>(copy.size() - 1);
  float fraction =
    (percentile / 100.0 - percent_i) / ((index + 1.0) / (copy.size() - 1.0) - percent_i);
  float res = copy[index] + fraction * (copy[index + 1] - copy[index]);
  return res;
}

} // namespace

int entry(int argc, char **argv)
{
  arser::Arser arser("minmax-embedder embeds given minmax into circle");
  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);
  // named args
  arser.add_argument("--min_percentile")
    .type(arser::DataType::FLOAT)
    .default_value(1.f)
    .help("Set min percentile");
  arser.add_argument("--max_percentile")
    .type(arser::DataType::FLOAT)
    .default_value(99.f)
    .help("Set max percentile");
  arser.add_argument("-o").default_value("out.circle").help("Path to output circle model");
  // positional args: minmax(h5), input(circle)
  arser.add_argument("circle").help("Path to input circle model");
  arser.add_argument("minmax").help("Path to minmax data in hdf5");
  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  std::string mm_path = arser.get<std::string>("minmax");
  std::string ic_path = arser.get<std::string>("circle");
  std::string oc_path = arser.get<std::string>("-o");
  float min_percentile = arser.get<float>("--min_percentile");
  float max_percentile = arser.get<float>("--max_percentile");

  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(ic_path);
  if (module.get() == nullptr)
    return EXIT_FAILURE;

  minmax::h5::Reader mmr{mm_path};

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    /* read subgraph inputs */
    const auto input_nodes = loco::input_nodes(graph);
    const auto num_inputs = input_nodes.size();
    for (uint32_t input_idx = 0; input_idx < num_inputs; ++input_idx)
    {
      const auto *circle_input = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      if (not circle_input->index() == input_idx)
      {
        throw std::runtime_error("The input order in model and recording are different.");
      }
      auto minmax = mmr.read_input(0, idx, input_idx);
      auto min = getNthPercentile(minmax.min_vector, min_percentile);
      auto max = getNthPercentile(minmax.max_vector, max_percentile);
      auto quantparam = std::make_unique<luci::CircleQuantParam>();
      quantparam->min.push_back(min);
      quantparam->max.push_back(max);
      const auto *circle_node = loco::must_cast<const luci::CircleNode *>(input_nodes[input_idx]);
      auto mutable_node = const_cast<luci::CircleNode *>(circle_node);
      mutable_node->quantparam(std::move(quantparam));
    }

    /* read op outputs */
    uint32_t node_num = graph->nodes()->size();
    for (uint32_t i = 0; i < node_num; ++i)
    {
      auto node = loco::must_cast<luci::CircleNode *>(graph->nodes()->at(i));
      if (not has_node_id(node)) // Skip non-op nodes (e.g. input/const/output)
        continue;
      auto op_idx = luci::get_node_id(node);
      auto minmax = mmr.read(0, idx, op_idx);
      auto min = getNthPercentile(minmax.min_vector, min_percentile);
      auto max = getNthPercentile(minmax.max_vector, max_percentile);
      auto quantparam = std::make_unique<luci::CircleQuantParam>();
      quantparam->min.push_back(min);
      quantparam->max.push_back(max);
      auto mutable_node = const_cast<luci::CircleNode *>(node);
      mutable_node->quantparam(std::move(quantparam));
    }

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return 255;
    }
  }

  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(module.get(), oc_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << oc_path << "'" << std::endl;
    return 255;
  }

  return 0;
}
