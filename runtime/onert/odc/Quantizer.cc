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

extern "C" onert::odc::IQuantizer *create_quantizer() { return new onert::odc::Quantizer(); }
extern "C" void destroy_quantizer(onert::odc::IQuantizer *quantizer) { delete quantizer; }

namespace onert
{
namespace odc
{

using QuantizerOptions = luci::CircleQuantizer::Options;
using QuantizeType = onert::odc::QuantizeType;

namespace
{

void fillQuantizeOptionParam(QuantizerOptions *options, QuantizeType qtype)
{
  std::string output_type = "";
  switch (qtype)
  {
    case QuantizeType::ODC_QTYPE_U8_ASYM:
      output_type = "uint8";
      break;
    case QuantizeType::ODC_QTYPE_I16_SYM:
      output_type = "int16";
      break;
    case QuantizeType::ODC_QTYPE_WO_I8_SYM:
      output_type = "wo_int8";
      break;
    case QuantizeType::ODC_QTYPE_WO_I16_SYM:
      output_type = "wo_int16";
      break;
    default:
      throw std::runtime_error("Invalid quantization type");
  }
  options->param(QuantizerOptions::AlgorithmParameters::Quantize_output_model_dtype, output_type);
  options->param(QuantizerOptions::AlgorithmParameters::Quantize_input_model_dtype, "float32");
  options->param(QuantizerOptions::AlgorithmParameters::Quantize_granularity, "channel");

  if (qtype == QuantizeType::ODC_QTYPE_U8_ASYM)
  {
    options->param(QuantizerOptions::AlgorithmParameters::Quantize_input_type, "uint8");
    options->param(QuantizerOptions::AlgorithmParameters::Quantize_output_type, "uint8");
  }
  else if (qtype == QuantizeType::ODC_QTYPE_I16_SYM)
  {
    options->param(QuantizerOptions::AlgorithmParameters::Quantize_input_type, "int16");
    options->param(QuantizerOptions::AlgorithmParameters::Quantize_output_type, "int16");
  }
}

} // namespace

int Quantizer::quantize(const char *in, const char *out, QuantizeType qtype)
{
  if (not in || not out)
    return 1;

  bool full_quantize = false;
  if (qtype == QuantizeType::ODC_QTYPE_U8_ASYM || qtype == QuantizeType::ODC_QTYPE_I16_SYM)
    full_quantize = true;

  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(std::string(in));
  if (module.get() == nullptr)
    return 1;

  luci::CircleQuantizer quantizer;
  auto options = quantizer.options();
  if (options == nullptr)
    return 1;

  // Fill quantization type param
  fillQuantizeOptionParam(options, qtype);

  // Additional phase for full quantization
  if (full_quantize)
  {
    // Fake quantization
    options->enable(QuantizerOptions::Algorithm::QuantizeDequantizeWeights);
    for (size_t idx = 0; idx < module->size(); ++idx)
    {
      auto graph = module->graph(idx);

      // quantize the graph
      quantizer.quantize(graph);
    }

    // TODO Record minmax by minmax-embedder
    throw std::runtime_error{"Not implemented yet"};
  }

  if (full_quantize)
    options->enable(QuantizerOptions::Algorithm::QuantizeWithMinMax);
  else
    options->enable(QuantizerOptions::Algorithm::QuantizeWeights);

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

} // namespace odc
} // namespace onert
