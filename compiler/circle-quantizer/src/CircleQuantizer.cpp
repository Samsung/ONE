/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci/ImporterEx.h>
#include <luci/CircleQuantizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>

#include <oops/InternalExn.h>
#include <arser/arser.h>
#include <vconone/vconone.h>
#include <json.h>

#include <iostream>
#include <map>
#include <string>

using LayerParam = luci::CircleQuantizer::Options::LayerParam;
using LayerParams = luci::CircleQuantizer::Options::LayerParams;
using LayerParamsSet = luci::CircleQuantizer::Options::LayerParamsSet;
using Algorithms = luci::CircleQuantizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleQuantizer::Options::AlgorithmParameters;

struct QConfReader
{
  void init(const std::string &filename)
  {
    std::ifstream ifs;

    ifs.open(filename);

    // Failed to open cfg file
    if (not ifs.is_open())
      throw std::runtime_error("Cannot open config file. " + filename);

    JSONCPP_STRING errs;
    Json::CharReaderBuilder builder;

    // Failed to parse
    if (not parseFromStream(builder, ifs, &_root, &errs))
      throw std::runtime_error("Cannot parse config file (json format). " + errs);

    ifs.close();
  }

  Json::Value &root(void) { return _root; }

private:
  Json::Value _root;
};

LayerParams read_layer_params(std::string &filename)
{
  QConfReader qcr;
  qcr.init(filename);

  auto layers = qcr.root()["layers"];
  LayerParams p;
  for (auto layer : layers)
  {
    if (layer.isMember("name"))
    {
      auto l = std::make_shared<LayerParam>();
      {
        l->name = layer["name"].asString();
        l->dtype = layer["dtype"].asString();
        l->granularity = layer["granularity"].asString();
      }
      p.emplace_back(l);
    }

    // Multiple names with the same dtype & granularity
    if (layer.isMember("names"))
    {
      for (auto name : layer["names"])
      {
        auto l = std::make_shared<LayerParam>();
        {
          l->name = name.asString();
          l->dtype = layer["dtype"].asString();
          l->granularity = layer["granularity"].asString();
        }
        p.emplace_back(l);
      }
    }
  }

  return p;
}

LayerParamsSet read_layer_params_set(std::string &filename)
{
  LayerParamsSet lpss;

  // read default values
  LayerParams lps = read_layer_params(filename);
  lpss.emplace_back(lps);

  QConfReader qcr;
  qcr.init(filename);

  auto layers = qcr.root()["layers"];
  // alternate names
  for (auto layer : layers)
  {
    const std::string key_alt_names = "alternate";
    if (layer.isMember(key_alt_names))
    {
      auto alternate = layer[key_alt_names];
      for (auto altkey : alternate.getMemberNames())
      {
        LayerParams lps;
        for (auto altvalue : alternate[altkey])
        {
          auto l = std::make_shared<LayerParam>();
          {
            l->name = altvalue.asString();
            l->dtype = layer["dtype"].asString();
            l->granularity = layer["granularity"].asString();
          }
          lps.emplace_back(l);
        }
        lpss.emplace_back(lps);
      }
    }
  }

  return lpss;
}

void print_exclusive_options(void)
{
  std::cout << "Use only one of the 3 options below." << std::endl;
  std::cout << "    --quantize_dequantize_weights" << std::endl;
  std::cout << "    --quantize_with_minmax" << std::endl;
  std::cout << "    --requantize" << std::endl;
  std::cout << "    --force_quantparam" << std::endl;
  std::cout << "    --quantize_weights" << std::endl;
}

void print_version(void)
{
  std::cout << "circle-quantizer version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  luci::CircleQuantizer quantizer;

  auto options = quantizer.options();
  auto settings = luci::UserSettings::settings();

  const std::string qdqw = "--quantize_dequantize_weights";
  const std::string qwmm = "--quantize_with_minmax";
  const std::string rq = "--requantize";
  const std::string fq = "--force_quantparam";
  const std::string cq = "--copy_quantparam";
  const std::string fake_quant = "--fake_quantize";
  const std::string qw = "--quantize_weights";
  const std::string cfg = "--config";

  const std::string tf_maxpool = "--TF-style_maxpool";

  const std::string gpd = "--generate_profile_data";

  const std::string save_min_max = "--save_min_max";

  arser::Arser arser("circle-quantizer provides circle model quantization");

  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);

  arser.add_argument(qdqw)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .help("Quantize-dequantize weight values required action before quantization. "
          "Three arguments required: input_model_dtype(float32) "
          "output_model_dtype(uint8) granularity(layer, channel)");

  arser.add_argument(qwmm)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .help("Quantize with min/max values. "
          "Three arguments required: input_model_dtype(float32) "
          "output_model_dtype(uint8) granularity(layer, channel)");

  arser.add_argument(tf_maxpool)
    .nargs(0)
    .default_value(false)
    .help("Force MaxPool Op to have the same input/output quantparams. NOTE: This feature can "
          "degrade accuracy of some models");

  arser.add_argument(save_min_max)
    .nargs(0)
    .default_value(false)
    .help("Save recorded min/max values.");

  arser.add_argument(fake_quant)
    .nargs(0)
    .help("Convert a quantized model to a fake-quantized model. NOTE: This feature will "
          "generate an fp32 model.");

  arser.add_argument(rq)
    .nargs(2)
    .type(arser::DataType::STR_VEC)
    .help("Requantize a quantized model. "
          "Two arguments required: input_model_dtype(int8) "
          "output_model_dtype(uint8)");

  arser.add_argument(fq)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .accumulated(true)
    .help("Write quantization parameters to the specified tensor. "
          "Three arguments required: tensor_name(string), "
          "scale(float) zero_point(int)");

  arser.add_argument(cq)
    .nargs(2)
    .type(arser::DataType::STR_VEC)
    .accumulated(true)
    .help("Copy quantization parameter from a tensor to another tensor."
          "Two arguments required: source_tensor_name(string), "
          "destination_tensor_name(string)");

  arser.add_argument(qw)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .help("Quantize weights values only"
          "Three arguments required: input_model_dtype(float32) "
          "output_model_dtype(int8, int16) granularity(channel)");

  arser.add_argument("--input_type")
    .help("Input type of quantized model (uint8, int16, int32, int64, float32, or bool). For "
          "multiple inputs, "
          "use comma-separated values. e.g., uint8,int16");

  arser.add_argument("--output_type")
    .help("Output type of quantized model (uint8, int16, int32, int64, float32, or bool). For "
          "multiple outputs, "
          "use comma-separated values. e.g., uint8,int16");

  arser.add_argument(cfg).help("Path to the quantization configuration file");

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output").help("Output circle model");

  arser.add_argument(gpd).nargs(0).required(false).default_value(false).help(
    "This will turn on profiling data generation.");

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

  {
    // only one of qdqw, qwmm, rq, fq, cq, fake_quant, qw option can be used
    int32_t opt_used = arser[qdqw] ? 1 : 0;
    opt_used += arser[qwmm] ? 1 : 0;
    opt_used += arser[rq] ? 1 : 0;
    opt_used += arser[fq] ? 1 : 0;
    opt_used += arser[cq] ? 1 : 0;
    opt_used += arser[fake_quant] ? 1 : 0;
    opt_used += arser[qw] ? 1 : 0;
    if (opt_used != 1)
    {
      print_exclusive_options();
      return 255;
    }
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  if (arser[qdqw])
  {
    auto values = arser.get<std::vector<std::string>>(qdqw);
    if (values.size() != 3)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::QuantizeDequantizeWeights);

    options->param(AlgorithmParameters::Quantize_input_model_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_model_dtype, values.at(1));
    options->param(AlgorithmParameters::Quantize_granularity, values.at(2));

    if (arser[cfg])
    {
      auto filename = arser.get<std::string>(cfg);
      try
      {
        auto layer_params = read_layer_params(filename);

        options->layer_params(AlgorithmParameters::Quantize_layer_params, layer_params);

        auto layer_params_set = read_layer_params_set(filename);

        options->layer_params_set(layer_params_set);
      }
      catch (const std::runtime_error &e)
      {
        std::cerr << e.what() << '\n';
        return 255;
      }
    }
  }

  if (arser[qwmm])
  {
    auto values = arser.get<std::vector<std::string>>(qwmm);
    if (values.size() != 3)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::QuantizeWithMinMax);

    options->param(AlgorithmParameters::Quantize_input_model_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_model_dtype, values.at(1));
    options->param(AlgorithmParameters::Quantize_granularity, values.at(2));

    if (arser["--input_type"])
      options->param(AlgorithmParameters::Quantize_input_type,
                     arser.get<std::string>("--input_type"));

    if (arser["--output_type"])
      options->param(AlgorithmParameters::Quantize_output_type,
                     arser.get<std::string>("--output_type"));

    if (arser[tf_maxpool] and arser.get<bool>(tf_maxpool))
      options->param(AlgorithmParameters::Quantize_TF_style_maxpool, "True");

    if (arser[save_min_max] and arser.get<bool>(save_min_max))
      options->param(AlgorithmParameters::Quantize_save_min_max, "True");

    if (arser[cfg])
    {
      auto filename = arser.get<std::string>(cfg);
      try
      {
        auto layer_params = read_layer_params(filename);

        options->layer_params(AlgorithmParameters::Quantize_layer_params, layer_params);

        auto layer_params_set = read_layer_params_set(filename);

        options->layer_params_set(layer_params_set);
      }
      catch (const std::runtime_error &e)
      {
        std::cerr << e.what() << '\n';
        return 255;
      }
    }
  }

  if (arser[rq])
  {
    auto values = arser.get<std::vector<std::string>>(rq);
    if (values.size() != 2)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::Requantize);

    options->param(AlgorithmParameters::Quantize_input_model_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_model_dtype, values.at(1));
  }

  if (arser[fq])
  {
    auto values = arser.get<std::vector<std::vector<std::string>>>(fq);

    std::vector<std::string> tensors;
    std::vector<std::string> scales;
    std::vector<std::string> zero_points;

    for (auto const value : values)
    {
      if (value.size() != 3)
      {
        std::cerr << arser;
        return 255;
      }

      tensors.push_back(value[0]);
      scales.push_back(value[1]);
      zero_points.push_back(value[2]);
    }

    options->enable(Algorithms::ForceQuantParam);

    options->params(AlgorithmParameters::Quantize_tensor_names, tensors);
    options->params(AlgorithmParameters::Quantize_scales, scales);
    options->params(AlgorithmParameters::Quantize_zero_points, zero_points);
  }

  if (arser[cq])
  {
    auto values = arser.get<std::vector<std::vector<std::string>>>(cq);

    std::vector<std::string> src;
    std::vector<std::string> dst;

    for (auto const value : values)
    {
      if (value.size() != 2)
      {
        std::cerr << arser;
        return 255;
      }

      src.push_back(value[0]);
      dst.push_back(value[1]);
    }

    options->enable(Algorithms::CopyQuantParam);

    options->params(AlgorithmParameters::Quantize_src_tensor_names, src);
    options->params(AlgorithmParameters::Quantize_dst_tensor_names, dst);
  }

  if (arser[fake_quant])
    options->enable(Algorithms::ConvertToFakeQuantizedModel);

  if (arser[qw])
  {
    auto values = arser.get<std::vector<std::string>>(qw);
    if (values.size() != 3)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::QuantizeWeights);

    options->param(AlgorithmParameters::Quantize_input_model_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_model_dtype, values.at(1));
    options->param(AlgorithmParameters::Quantize_granularity, values.at(2));
  }

  std::string input_path = arser.get<std::string>("input");
  std::string output_path = arser.get<std::string>("output");

  if (arser[gpd])
    settings->set(luci::UserSettings::Key::ProfilingDataGen, true);

  // Load model from the file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(input_path);
  if (module.get() == nullptr)
    return EXIT_FAILURE;

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    // quantize the graph
    quantizer.quantize(graph);

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return 255;
    }
  }

  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(module.get(), output_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
    return 255;
  }

  return 0;
}
