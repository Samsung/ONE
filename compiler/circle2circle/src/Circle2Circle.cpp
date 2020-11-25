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

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/CircleOptimizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>

#include <oops/InternalExn.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <functional>
#include <iostream>
#include <string>

using Algorithms = luci::CircleOptimizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleOptimizer::Options::AlgorithmParameters;

void print_version(void)
{
  std::cout << "circle2circle version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  // Simple argument parser (based on map)
  luci::CircleOptimizer optimizer;

  auto options = optimizer.options();
  auto settings = luci::UserSettings::settings();

  arser::Arser arser("circle2circle provides circle model optimization and transformations");

  arser.add_argument("--version")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("Show version information and exit")
      .exit_with(print_version);

  arser.add_argument("--all").nargs(0).required(false).default_value(false).help(
      "Enable all optimize options");

  arser.add_argument("--fold_dequantize")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fold dequantize op");

  arser.add_argument("--fuse_activation_function")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse Activation function to a preceding operator");

  arser.add_argument("--fuse_add_with_tconv")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse Add operator to Transposed Convolution operator");

  arser.add_argument("--fuse_batchnorm_with_tconv")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse BatchNorm operators to Transposed Convolution operator");

  arser.add_argument("--fuse_bcq")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse operators and apply Binary Coded Quantization");

  arser.add_argument("--fuse_instnorm")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse operators to InstanceNorm operator");

  arser.add_argument("--make_batchnorm_gamma_positive")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will make negative gamma of BatchNorm into a small positive value (1e-10). Note "
            "that this pass can change the execution result of the model. So, use it only when the "
            "impact is known to be acceptable.");

  arser.add_argument("--fuse_preactivation_batchnorm")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse BatchNorm operators of pre-activations to Convolution operator");

  arser.add_argument("--remove_redundant_transpose")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will fuse or remove subsequent Transpose operators");

  arser.add_argument("--resolve_customop_add")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will convert Custom(Add) to Add operator");

  arser.add_argument("--resolve_customop_batchmatmul")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will convert Custom(BatchMatmul) to BatchMatmul operator");

  arser.add_argument("--resolve_customop_matmul")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will convert Custom(Matmul) to Matmul operator");

  arser.add_argument("--shuffle_weight_to_16x1float32")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will convert weight format of FullyConnected to SHUFFLED16x1FLOAT32. Note that "
            "it only converts weights whose row is a multiple of 16");

  arser.add_argument("--substitute_pack_to_reshape")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will convert single input Pack to Reshape");

  arser.add_argument("--mute_warnings")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will turn off warning messages");

  arser.add_argument("--disable_validation")
      .nargs(0)
      .required(false)
      .default_value(false)
      .help("This will turn off operator validations. May help input model investigation.");

  arser.add_argument("input").nargs(1).type(arser::DataType::STR).help("Input circle model");
  arser.add_argument("output").nargs(1).type(arser::DataType::STR).help("Output circle model");

  // sparsification argument
  arser.add_argument("--sparsify_tensor")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(false)
      .help("Tensor name that you want to sparsify");

  arser.add_argument("--sparsify_traversal_order")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(false)
      .default_value("0,1,2,3")
      .help("Traversal order of dimensions. Default value: 0,1,2,3");

  arser.add_argument("--sparsify_format")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(false)
      .default_value("d,s")
      .help("Format of each dimension. 'd' stands for dense, 's' stands for sparse(CSR). Default "
            "value: d,s");

  arser.add_argument("--sparsify_block_size")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(false)
      .help("Size of each block dimension");

  arser.add_argument("--sparsify_block_map")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(false)
      .default_value("0,1")
      .help("Map from block dimension to the original tensor dimension. Default value: 0,1");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  if (arser.get<bool>("--all"))
  {
    options->enable(Algorithms::FuseBCQ);
    options->enable(Algorithms::FuseInstanceNorm);
    options->enable(Algorithms::ResolveCustomOpAdd);
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
    options->enable(Algorithms::ResolveCustomOpMatMul);
    options->enable(Algorithms::RemoveRedundantTranspose);
    options->enable(Algorithms::SubstitutePackToReshape);
  }
  if (arser.get<bool>("--fold_dequantize"))
    options->enable(Algorithms::FoldDequantize);
  if (arser.get<bool>("--fuse_activation_function"))
    options->enable(Algorithms::FuseActivationFunction);
  if (arser.get<bool>("--fuse_add_with_tconv"))
    options->enable(Algorithms::FuseAddWithTConv);
  if (arser.get<bool>("--fuse_batchnorm_with_tconv"))
    options->enable(Algorithms::FuseBatchNormWithTConv);
  if (arser.get<bool>("--fuse_bcq"))
    options->enable(Algorithms::FuseBCQ);
  if (arser.get<bool>("--fuse_instnorm"))
    options->enable(Algorithms::FuseInstanceNorm);
  if (arser.get<bool>("--make_batchnorm_gamma_positive"))
    options->enable(Algorithms::MakeBatchNormGammaPositive);
  if (arser.get<bool>("--fuse_preactivation_batchnorm"))
    options->enable(Algorithms::FusePreActivationBatchNorm);
  if (arser.get<bool>("--remove_redundant_transpose"))
    options->enable(Algorithms::RemoveRedundantTranspose);
  if (arser.get<bool>("--resolve_customop_add"))
    options->enable(Algorithms::ResolveCustomOpAdd);
  if (arser.get<bool>("--resolve_customop_batchmatmul"))
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
  if (arser.get<bool>("--resolve_customop_matmul"))
    options->enable(Algorithms::ResolveCustomOpMatMul);
  if (arser.get<bool>("--shuffle_weight_to_16x1float32"))
    options->enable(Algorithms::ShuffleWeightTo16x1Float32);
  if (arser.get<bool>("--substitute_pack_to_reshape"))
    options->enable(Algorithms::SubstitutePackToReshape);
  if (arser.get<bool>("--mute_warnings"))
    settings->set(luci::UserSettings::Key::MuteWarnings, true);
  if (arser.get<bool>("--disable_validation"))
    settings->set(luci::UserSettings::Key::DisableValidation, true);

  std::string input_path = arser.get<std::string>("input");
  std::string output_path = arser.get<std::string>("output");

  if (arser["--sparsify_tensor"])
  {
    options->enable(Algorithms::SparsifyTensorPass);
    options->param(AlgorithmParameters::Sparsify_tensor_name,
                   arser.get<std::string>("--sparsify_tensor"));
    options->param(AlgorithmParameters::Sparsify_traversal_order,
                   arser.get<std::string>("--sparsify_traversal_order"));
    options->param(AlgorithmParameters::Sparsify_format,
                   arser.get<std::string>("--sparsify_format"));
    if (arser["--sparsify_block_size"])
      options->param(AlgorithmParameters::Sparsify_block_size,
                     arser.get<std::string>("--sparsify_block_size"));
    else
    {
      std::cerr << "ERROR: Block size not provided" << std::endl;
      return 255;
    }
    options->param(AlgorithmParameters::Sparsify_block_map,
                   arser.get<std::string>("--sparsify_block_map"));
  }

  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data;

  try
  {
    model_data = file_loader.load();
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }

  flatbuffers::Verifier verifier{reinterpret_cast<uint8_t *>(model_data.data()), model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cerr << "ERROR: Invalid input file '" << input_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  const circle::Model *circle_model = circle::GetModel(model_data.data());
  if (circle_model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << input_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // Import from input Circle file
  luci::Importer importer;
  auto module = importer.importModule(circle_model);

  // call luci optimizations for module
  optimizer.optimize(module.get());

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    // call luci optimizations for graph
    optimizer.optimize(graph);
    optimizer.sparsify(graph);

    if (!luci::validate(graph))
    {
      if (settings->get(luci::UserSettings::Key::DisableValidation))
        std::cerr << "WARNING: Optimized graph is invalid" << std::endl;
      else
      {
        std::cerr << "ERROR: Optimized graph is invalid" << std::endl;
        return 255;
      }
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
