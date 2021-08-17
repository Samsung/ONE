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
#include <luci/Service/ChangeOutputs.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>

#include <oops/InternalExn.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

using Algorithms = luci::CircleOptimizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleOptimizer::Options::AlgorithmParameters;

void print_version(void)
{
  std::cout << "circle2circle version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

void csv_tokenize(const std::string &data, std::vector<std::string> &result)
{
  const char delim = ',';
  std::string token;
  std::stringstream ss(data);

  while (std::getline(ss, token, delim))
    result.push_back(token);
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

  arser.add_argument("-V", "--verbose")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("output additional information to stdout or stderr");

  arser.add_argument("--O1").nargs(0).required(false).default_value(false).help(
    "Enable O1 optimize options");

  arser.add_argument("--fold_add_v2")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fold AddV2 operators with constant inputs");

  arser.add_argument("--fold_cast")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fold Cast operators with constant input");

  arser.add_argument("--fold_dequantize")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fold dequantize op");

  arser.add_argument("--fold_sparse_to_dense")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fold SparseToDense operator");

  arser.add_argument("--forward_reshape_to_unaryop")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will move Reshape after UnaryOp for centain condition");

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

  arser.add_argument("--fuse_batchnorm_with_conv")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fuse BatchNorm operators to Convolution operator");

  arser.add_argument("--fuse_batchnorm_with_dwconv")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fuse BatchNorm operators to Depthwise Convolution operator");

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

  arser.add_argument("--fuse_mean_with_mean")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fuse two Mean operations when they follow one by one."
          "This will fold them into one operation and merge reduction indices.");

  arser.add_argument("--fuse_transpose_with_mean")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fuse Mean operation with a preceding Transpose under certain conditions.");

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

  arser.add_argument("--remove_fakequant")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will remove FakeQuant operators");

  arser.add_argument("--remove_quantdequant")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will remove Quantize-Dequantize sequence");

  arser.add_argument("--remove_redundant_reshape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fuse or remove subsequent Reshape operators");

  arser.add_argument("--remove_redundant_transpose")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will fuse or remove subsequent Transpose operators");

  arser.add_argument("--remove_unnecessary_reshape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will remove unnecessary reshape operators");

  arser.add_argument("--remove_unnecessary_slice")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will remove unnecessary slice operators");

  arser.add_argument("--remove_unnecessary_strided_slice")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will remove unnecessary strided slice operators");

  arser.add_argument("--remove_unnecessary_split")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will remove unnecessary split operators");

  arser.add_argument("--replace_cw_mul_add_with_depthwise_conv")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will replace channel-wise mul/add with DepthwiseConv2D operator");

  arser.add_argument("--replace_sub_with_add")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will replace sub with add operator");

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

  arser.add_argument("--substitute_squeeze_to_reshape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will convert certain condition Squeeze to Reshape");

  arser.add_argument("--substitute_strided_slice_to_reshape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will convert certain condition Strided_Slice to Reshape");

  arser.add_argument("--substitute_transpose_to_reshape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will convert single input Transpose to Reshape");

  arser.add_argument("--convert_nchw_to_nhwc")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Experimental: This will convert NCHW operators to NHWC under the assumption that "
          "input model is NCHW.");

  arser.add_argument("--nchw_to_nhwc_input_shape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Convert the input shape of the model (argument for --convert_nchw_to_nhwc).");

  arser.add_argument("--nchw_to_nhwc_output_shape")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Convert the output shape of the model (argument for --convert_nchw_to_nhwc).");

  arser.add_argument("--transform_min_max_to_relu6")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Transform Minimum(6)-Maximum(0) pattern to Relu6 operator");

  arser.add_argument("--transform_min_relu_to_relu6")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Transform Minimum(6)-Relu pattern to Relu6 operator");

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

  arser.add_argument("--generate_profile_data")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("This will turn on profiling data generation.");

  arser.add_argument("--change_outputs")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Experimental: Change first subgraph output nodes to CSV names");

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
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  if (arser.get<bool>("--verbose"))
    setenv("LUCI_LOG", "100", true);
  else
    setenv("LUCI_LOG", "0", true);

  if (arser.get<bool>("--O1"))
  {
    options->enable(Algorithms::FuseBCQ);
    options->enable(Algorithms::FuseInstanceNorm);
    options->enable(Algorithms::ResolveCustomOpAdd);
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
    options->enable(Algorithms::ResolveCustomOpMatMul);
    options->enable(Algorithms::RemoveRedundantTranspose);
    options->enable(Algorithms::SubstitutePackToReshape);
  }
  if (arser.get<bool>("--fold_add_v2"))
    options->enable(Algorithms::FoldAddV2);
  if (arser.get<bool>("--fold_cast"))
    options->enable(Algorithms::FoldCast);
  if (arser.get<bool>("--fold_dequantize"))
    options->enable(Algorithms::FoldDequantize);
  if (arser.get<bool>("--fold_sparse_to_dense"))
    options->enable(Algorithms::FoldSparseToDense);
  if (arser.get<bool>("--forward_reshape_to_unaryop"))
    options->enable(Algorithms::ForwardReshapeToUnaryOp);
  if (arser.get<bool>("--fuse_activation_function"))
    options->enable(Algorithms::FuseActivationFunction);
  if (arser.get<bool>("--fuse_batchnorm_with_conv"))
    options->enable(Algorithms::FuseBatchNormWithConv);
  if (arser.get<bool>("--fuse_add_with_tconv"))
    options->enable(Algorithms::FuseAddWithTConv);
  if (arser.get<bool>("--fuse_batchnorm_with_dwconv"))
    options->enable(Algorithms::FuseBatchNormWithDwConv);
  if (arser.get<bool>("--fuse_batchnorm_with_tconv"))
    options->enable(Algorithms::FuseBatchNormWithTConv);
  if (arser.get<bool>("--fuse_bcq"))
    options->enable(Algorithms::FuseBCQ);
  if (arser.get<bool>("--fuse_instnorm"))
    options->enable(Algorithms::FuseInstanceNorm);
  if (arser.get<bool>("--fuse_mean_with_mean"))
    options->enable(Algorithms::FuseMeanWithMean);
  if (arser.get<bool>("--make_batchnorm_gamma_positive"))
    options->enable(Algorithms::MakeBatchNormGammaPositive);
  if (arser.get<bool>("--fuse_preactivation_batchnorm"))
    options->enable(Algorithms::FusePreActivationBatchNorm);
  if (arser.get<bool>("--fuse_transpose_with_mean"))
    options->enable(Algorithms::FuseTransposeWithMean);
  if (arser.get<bool>("--remove_fakequant"))
    options->enable(Algorithms::RemoveFakeQuant);
  if (arser.get<bool>("--remove_quantdequant"))
    options->enable(Algorithms::RemoveQuantDequantSeq);
  if (arser.get<bool>("--remove_redundant_reshape"))
    options->enable(Algorithms::RemoveRedundantReshape);
  if (arser.get<bool>("--remove_redundant_transpose"))
    options->enable(Algorithms::RemoveRedundantTranspose);
  if (arser.get<bool>("--remove_unnecessary_reshape"))
    options->enable(Algorithms::RemoveUnnecessaryReshape);
  if (arser.get<bool>("--remove_unnecessary_slice"))
    options->enable(Algorithms::RemoveUnnecessarySlice);
  if (arser.get<bool>("--remove_unnecessary_strided_slice"))
    options->enable(Algorithms::RemoveUnnecessaryStridedSlice);
  if (arser.get<bool>("--remove_unnecessary_split"))
    options->enable(Algorithms::RemoveUnnecessarySplit);
  if (arser.get<bool>("--replace_cw_mul_add_with_depthwise_conv"))
    options->enable(Algorithms::ReplaceMulAddWithDepthwiseConv);
  if (arser.get<bool>("--replace_sub_with_add"))
    options->enable(Algorithms::ReplaceSubWithAdd);
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
  if (arser.get<bool>("--substitute_squeeze_to_reshape"))
    options->enable(Algorithms::SubstituteSqueezeToReshape);
  if (arser.get<bool>("--substitute_strided_slice_to_reshape"))
    options->enable(Algorithms::SubstituteStridedSliceToReshape);
  if (arser.get<bool>("--substitute_transpose_to_reshape"))
    options->enable(Algorithms::SubstituteTransposeToReshape);
  if (arser.get<bool>("--transform_min_max_to_relu6"))
    options->enable(Algorithms::TransformMinMaxToRelu6Pass);
  if (arser.get<bool>("--transform_min_relu_to_relu6"))
    options->enable(Algorithms::TransformMinReluToRelu6Pass);

  if (arser.get<bool>("--mute_warnings"))
    settings->set(luci::UserSettings::Key::MuteWarnings, true);
  if (arser.get<bool>("--disable_validation"))
    settings->set(luci::UserSettings::Key::DisableValidation, true);
  if (arser.get<bool>("--generate_profile_data"))
    settings->set(luci::UserSettings::Key::ProfilingDataGen, true);

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

  if (arser.get<bool>("--convert_nchw_to_nhwc"))
  {
    options->enable(Algorithms::ConvertNCHWToNHWC);
    if (arser.get<bool>("--nchw_to_nhwc_input_shape"))
      options->param(AlgorithmParameters::NCHW_to_NHWC_input_shape, "true");
    if (arser.get<bool>("--nchw_to_nhwc_output_shape"))
      options->param(AlgorithmParameters::NCHW_to_NHWC_output_shape, "true");
  }

  // Change output nodes
  bool change_outputs = false;
  std::vector<std::string> new_outputs;
  if (arser["--change_outputs"])
  {
    change_outputs = true;
    auto csv_nodes = arser.get<std::string>("--change_outputs");
    csv_tokenize(csv_nodes, new_outputs);
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

  if (change_outputs)
  {
    auto graph = module->graph(0);
    luci::change_outputs(graph, new_outputs);
  }

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
