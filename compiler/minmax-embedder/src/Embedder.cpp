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

#include "minmax-embedder/Embedder.h"

#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/ImporterEx.h>
#include <luci/IR/CircleNode.h>
#include <luci/IR/CircleQuantParam.h>
#include <luci/Profile/CircleNodeID.h>
#include <luci/Service/Validate.h>

#include "h5/Reader.h"

#include <cassert>
#include <cmath> // for std::floor
#include <iostream>
#include <string>

namespace
{

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

namespace minmax_embedder
{

void Embedder::embed(const std::string &output_path, const std::string &input_path,
                     const std::string &minmax_path, const EmbedderOptions &opt)
{
  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(input_path);
  if (module.get() == nullptr)
    throw std::runtime_error{"Input circle is invalid"};

  h5::Reader mmr{minmax_path};

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    /* read subgraph inputs */
    const auto input_nodes = loco::input_nodes(graph);
    const auto n_inputs = input_nodes.size();
    for (size_t input_idx = 0; input_idx < n_inputs; ++input_idx)
    {
      const auto *circle_input = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      if (circle_input->index() != input_idx)
        throw std::runtime_error("Input order in minmax recording does not match to circle");

      auto minmax = mmr.read_input(0, idx, input_idx);
      auto min = getNthPercentile(minmax.min_vector, opt.min_percentile);
      auto max = getNthPercentile(minmax.max_vector, opt.max_percentile);
      auto quantparam = std::make_unique<luci::CircleQuantParam>();
      quantparam->min.push_back(min);
      quantparam->max.push_back(max);
      const auto *circle_node = loco::must_cast<const luci::CircleNode *>(input_nodes[input_idx]);
      auto mutable_node = const_cast<luci::CircleNode *>(circle_node);
      mutable_node->quantparam(std::move(quantparam));
    }

    /* read op outputs */
    uint32_t n_nodes = graph->nodes()->size();
    for (uint32_t i = 0; i < n_nodes; ++i)
    {
      auto node = loco::must_cast<luci::CircleNode *>(graph->nodes()->at(i));
      if (not luci::has_node_id(node)) // Skip non-op nodes (e.g. input/const/output)
        continue;
      auto op_idx = luci::get_node_id(node);
      auto minmax = mmr.read(0, idx, op_idx);
      auto min = getNthPercentile(minmax.min_vector, opt.min_percentile);
      auto max = getNthPercentile(minmax.max_vector, opt.max_percentile);
      auto quantparam = std::make_unique<luci::CircleQuantParam>();
      quantparam->min.push_back(min);
      quantparam->max.push_back(max);
      auto mutable_node = const_cast<luci::CircleNode *>(node);
      mutable_node->quantparam(std::move(quantparam));
    }

    if (!luci::validate(graph))
      throw std::runtime_error{"Circle after embedding minmax is invalid"};
  }

  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(module.get(), output_path);

  if (!exporter.invoke(&contract))
    throw std::runtime_error{"Failed to export circle"};
}

} // namespace minmax_embedder
