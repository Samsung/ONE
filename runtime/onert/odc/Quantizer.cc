/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Quantizer.h"

#include <luci/ImporterEx.h>
#include <luci/CircleQuantizer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include <iostream>

int quantize(const char *in, const char *out, bool is_q16)
{
  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(std::string(in));
  if (module.get() == nullptr)
    return 1;

  luci::CircleQuantizer quantizer;
  auto options = quantizer.options();
  {
    options->enable(luci::CircleQuantizer::Options::Algorithm::QuantizeWeights);

    using AlgorithmParameters = luci::CircleQuantizer::Options::AlgorithmParameters;
    options->param(AlgorithmParameters::Quantize_input_model_dtype, "float32");
    options->param(AlgorithmParameters::Quantize_output_model_dtype, is_q16 ? "int16" : "int8");
    options->param(AlgorithmParameters::Quantize_granularity, "channel");
  }

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    // quantize the graph
    quantizer.quantize(graph);

    // Skip validate
    // TODO Validate if needed
#if 0
    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return 1;
    }
#endif
  }

  // Export to output Circle file
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(module.get(), std::string(out));

  if (!exporter.invoke(&contract))
    return 1;

  // Return 0 when luci::CircleQuantizer::Options::Algorithm::QuantizeWeights is ready
  return 0;
}
