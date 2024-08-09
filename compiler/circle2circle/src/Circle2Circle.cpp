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
#include <luci/CircleOptimizer.h>
#include <luci/DynamicBatchToSingleBatch.h>
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

void add_switch(arser::Arser &arser, const char *opt, const char *desc)
{
  arser.add_argument(opt).nargs(0).default_value(false).help(desc);
}

int entry(int argc, char **argv)
{
  // Simple argument parser (based on map)
  luci::CircleOptimizer optimizer;

  auto options = optimizer.options();
  auto settings = luci::UserSettings::settings();

  arser::Arser arser("circle2circle provides circle model optimization and transformations");

  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);

  add_switch(arser, "--fold_add_v2", "This will fold AddV2 operators with constant inputs");
  add_switch(arser, "--fold_cast", "This will fold Cast operators with constant input");
  add_switch(arser, "--fold_densify",
             "This will fold Densify operators with sparse constant input");
  add_switch(arser, "--fold_dequantize", "This will fold dequantize op");
  add_switch(arser, "--fold_dwconv",
             "This will fold Depthwise Convolution operator with constant inputs");
  add_switch(arser, "--fold_fully_connected",
             "This will fold FullyConnected operator with constant inputs");
  add_switch(arser, "--fold_gather", "This will fold Gather operator");
  add_switch(arser, "--fold_mul", "This will fold Mul operator");
  add_switch(arser, "--fold_reshape", "This will fold Reshape operator");
  add_switch(arser, "--fold_shape", "This will fold Shape operator");
  add_switch(arser, "--fold_sparse_to_dense", "This will fold SparseToDense operator");
  add_switch(arser, "--fold_squeeze", "This will fold Squeeze operator");
  add_switch(arser, "--forward_reshape_to_unaryop",
             "This will move Reshape after UnaryOp for centain condition");
  add_switch(arser, "--forward_transpose_op",
             "This will move Transpose Op forward if possible (for further optimization)");
  add_switch(arser, "--fuse_activation_function",
             "This will fuse Activation function to a preceding operator");
  add_switch(arser, "--fuse_horizontal_fc_layers",
             "This will fuse horizontal FullyConnected layers");
  add_switch(arser, "--fuse_add_to_fullyconnected_bias",
             "This will fuse Add to following FullyConnected bias");
  add_switch(arser, "--fuse_add_with_conv", "This will fuse Add operator to Convolution operator");
  add_switch(arser, "--fuse_add_with_fully_connected",
             "This will fuse Add operator to FullyConnected operator");
  add_switch(arser, "--fuse_add_with_tconv",
             "This will fuse Add operator to Transposed Convolution operator");
  add_switch(arser, "--fuse_batchnorm_with_conv",
             "This will fuse BatchNorm operators to Convolution operator");
  add_switch(arser, "--fuse_batchnorm_with_dwconv",
             "This will fuse BatchNorm operators to Depthwise Convolution operator");
  add_switch(arser, "--fuse_batchnorm_with_tconv",
             "This will fuse BatchNorm operators to Transposed Convolution operator");
  add_switch(arser, "--fuse_bcq", "This will fuse operators and apply Binary Coded Quantization");
  add_switch(arser, "--fuse_instnorm", "This will fuse operators to InstanceNorm operator");
  add_switch(arser, "--fuse_mean_with_mean",
             "This will fuse two Mean operations when they follow one by one. This will fold them "
             "into one operation and merge reduction indices.");
  add_switch(arser, "--fuse_mul_to_fullyconnected_weights",
             "This will fuse Mul to following FullyConnected weights");
  add_switch(arser, "--fuse_mul_with_conv",
             "This will fuse Mul operation with a preceding Conv if possible.");
  add_switch(arser, "--fuse_mul_with_div",
             "This will fuse Mul operation with a Div operation whose numerator is const.");
  add_switch(arser, "--fuse_slice_with_tconv",
             "This will fuse Slice operation with a preceding TConv if possible.");
  add_switch(arser, "--fuse_transpose_with_mean",
             "This will fuse Mean operation with a preceding Transpose under certain conditions.");
  add_switch(arser, "--make_batchnorm_gamma_positive",
             "This will make negative gamma of BatchNorm into a small positive value (1e-10). "
             "Note that this pass can change the execution result of the model. So, use it only "
             "when the impact is known to be acceptable.");
  add_switch(arser, "--fuse_preactivation_batchnorm",
             "This will fuse BatchNorm operators of pre-activations to Convolution operator");
  add_switch(arser, "--fuse_prelu", "This will fuse operators to PReLU operator");
  add_switch(arser, "--fuse_gelu", "This will fuse operators to GeLU operator");
  add_switch(arser, "--fuse_rsqrt", "This will fuse operators to Rsqrt operator");
  add_switch(arser, "--remove_duplicate_const", "This will remove all duplicate constant nodes");
  add_switch(arser, "--remove_fakequant", "This will remove FakeQuant operators");
  add_switch(arser, "--remove_gather_guard",
             "This will remove Add/FloorMod guards of Gather indices with certain conditions. "
             "CAUTION: user must guarantee that indices are all non-negative values.");
  add_switch(arser, "--remove_qdq_for_mpo",
             "This will remove QDQ to simulate mixed-precision operator");
  add_switch(arser, "--remove_quantdequant", "This will remove Quantize-Dequantize sequence");
  add_switch(arser, "--remove_redundant_quantize", "This will remove redundant Quantize operators");
  add_switch(arser, "--remove_redundant_reshape",
             "This will fuse or remove subsequent Reshape operators");
  add_switch(arser, "--remove_redundant_transpose",
             "This will fuse or remove subsequent Transpose operators");
  add_switch(arser, "--remove_unnecessary_add",
             "This will remove unnecessary add of zero constant");
  add_switch(arser, "--remove_unnecessary_reshape",
             "This will remove unnecessary reshape operators");
  add_switch(arser, "--remove_unnecessary_slice", "This will remove unnecessary slice operators");
  add_switch(arser, "--remove_unnecessary_strided_slice",
             "This will remove unnecessary strided slice operators");
  add_switch(arser, "--remove_unnecessary_split", "This will remove unnecessary split operators");
  add_switch(arser, "--remove_unnecessary_transpose",
             "This will remove unnecessary transpose operators");
  add_switch(arser, "--replace_cw_mul_add_with_depthwise_conv",
             "This will replace channel-wise mul/add with DepthwiseConv2D operator");
  add_switch(arser, "--replace_sub_with_add", "This will replace sub with add operator");
  add_switch(arser, "--replace_with_fc_gelu_fc",
             "This will replace a specific pattern into FC + Gelu + FC pattern.");
  add_switch(arser, "--resolve_customop_add", "This will convert Custom(Add) to Add operator");
  add_switch(arser, "--resolve_customop_batchmatmul",
             "This will convert Custom(BatchMatmul) to BatchMatmul operator");
  add_switch(arser, "--resolve_customop_matmul",
             "This will convert Custom(Matmul) to Matmul operator");
  add_switch(arser, "--resolve_customop_max_pool_with_argmax",
             "This will convert Custom(MaxPoolWithArgmax) to equivalent set of operators");
  add_switch(arser, "--resolve_customop_splitv",
             "This will convert Custom(SplitV) to SplitV operator");
  add_switch(arser, "--resolve_former_customop",
             "This will convert a former custom op to builtin in from schema version upgrade");
  add_switch(arser, "--shuffle_weight_to_16x1float32",
             "This will convert weight format of FullyConnected to SHUFFLED16x1FLOAT32. Note that "
             "it only converts weights whose row is a multiple of 16");
  add_switch(arser, "--replace_non_const_fc_with_batch_matmul",
             "Replace FullyConnected with BatchMatMul when its weight is non-constant");
  add_switch(arser, "--substitute_pack_to_reshape",
             "This will convert single input Pack to Reshape");
  add_switch(arser, "--substitute_padv2_to_pad",
             "This will convert certain condition PadV2 to Pad");
  add_switch(arser, "--substitute_splitv_to_split",
             "This will convert certain condition SplitV to Split operator");
  add_switch(arser, "--substitute_squeeze_to_reshape",
             "This will convert certain condition Squeeze to Reshape");
  add_switch(arser, "--substitute_strided_slice_to_reshape",
             "This will convert certain condition Strided_Slice to Reshape");
  add_switch(arser, "--substitute_transpose_to_reshape",
             "This will convert single input Transpose to Reshape");
  add_switch(arser, "--expand_broadcast_const", "This will expand broadcastable constant inputs");
  add_switch(arser, "--unroll_unidirseqlstm", "Unroll UnidirectionalSequenceLSTM operator.");
  add_switch(arser, "--compress_weights_huffman",
             "Loseless weights compression with Huffman encoding.");
  add_switch(arser, "--convert_nchw_to_nhwc",
             "Experimental: This will convert NCHW operators to NHWC under the assumption that "
             "input model is NCHW.");
  add_switch(arser, "--nchw_to_nhwc_input_shape",
             "Convert the input shape of the model (argument for --convert_nchw_to_nhwc).");
  add_switch(arser, "--nchw_to_nhwc_output_shape",
             "Convert the output shape of the model (argument for --convert_nchw_to_nhwc).");
  add_switch(arser, "--transform_min_max_to_relu6",
             "Transform Minimum(6)-Maximum(0) pattern to Relu6 operator");
  add_switch(arser, "--transform_min_relu_to_relu6",
             "Transform Minimum(6)-Relu pattern to Relu6 operator");
  add_switch(arser, "--transform_sqrt_div_to_rsqrt_mul",
             "Transform Sqrt-Div pattern to Rsqrt-Mul operators");
  add_switch(arser, "--decompose_hardswish",
             "Decompose HardSwish operator to Add, Mul and Relu6 operators");
  add_switch(arser, "--decompose_softmax",
             "Decompose Softmax operator into multiple operators for special backends");
  add_switch(arser, "--common_subexpression_elimination",
             "Perform common subexpression elimination");
  add_switch(arser, "--mute_warnings", "This will turn off warning messages");
  add_switch(arser, "--disable_validation",
             "This will turn off operator validations. May help input model investigation.");
  add_switch(arser, "--generate_profile_data", "This will turn on profiling data generation.");

  // NOTE Experimental options; these will be removed someday
  //      Add experimental options here
  add_switch(arser, "--exp_disable_sep_transposeconv_actfunc",
             "This will turn off experimental separation of activation function from "
             "TransposeConv.");

  // Convert dynamic batch to single batch
  // Users have to use this option only when the first dimension of rank 4 input (NHWC or NCHW)
  // is dynamic. Remove this comment after non-rank 4 is supported.
  add_switch(arser, "--dynamic_batch_to_single_batch",
             "Convert dynamic batch size (first dimension) of inputs to 1.");

  arser.add_argument("--change_outputs")
    .help("Experimental: Change first subgraph output nodes to CSV names");

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output").help("Output circle model");

  // sparsification argument
  arser.add_argument("--sparsify_tensor").help("Tensor name that you want to sparsify");

  arser.add_argument("--sparsify_traversal_order")
    .default_value("0,1,2,3")
    .help("Traversal order of dimensions. Default value: 0,1,2,3");

  arser.add_argument("--sparsify_format")
    .default_value("d,s")
    .help("Format of each dimension. 'd' stands for dense, 's' stands for sparse(CSR). Default "
          "value: d,s");

  arser.add_argument("--sparsify_block_size").help("Size of each block dimension");

  arser.add_argument("--sparsify_block_map")
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
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }
  if (arser.get<bool>("--fold_add_v2"))
    options->enable(Algorithms::FoldAddV2);
  if (arser.get<bool>("--fold_cast"))
    options->enable(Algorithms::FoldCast);
  if (arser.get<bool>("--fold_densify"))
    options->enable(Algorithms::FoldDensify);
  if (arser.get<bool>("--fold_dequantize"))
    options->enable(Algorithms::FoldDequantize);
  if (arser.get<bool>("--fold_dwconv"))
    options->enable(Algorithms::FoldDepthwiseConv2D);
  if (arser.get<bool>("--fold_fully_connected"))
    options->enable(Algorithms::FoldFullyConnected);
  if (arser.get<bool>("--fold_gather"))
    options->enable(Algorithms::FoldGather);
  if (arser.get<bool>("--fold_mul"))
    options->enable(Algorithms::FoldMul);
  if (arser.get<bool>("--fold_reshape"))
    options->enable(Algorithms::FoldReshape);
  if (arser.get<bool>("--fold_shape"))
    options->enable(Algorithms::FoldShape);
  if (arser.get<bool>("--fold_sparse_to_dense"))
    options->enable(Algorithms::FoldSparseToDense);
  if (arser.get<bool>("--fold_squeeze"))
    options->enable(Algorithms::FoldSqueeze);
  if (arser.get<bool>("--forward_reshape_to_unaryop"))
    options->enable(Algorithms::ForwardReshapeToUnaryOp);
  if (arser.get<bool>("--forward_transpose_op"))
    options->enable(Algorithms::ForwardTransposeOp);
  if (arser.get<bool>("--fuse_activation_function"))
    options->enable(Algorithms::FuseActivationFunction);
  if (arser.get<bool>("--fuse_horizontal_fc_layers"))
    options->enable(Algorithms::FuseHorizontalFullyConnected);
  if (arser.get<bool>("--fuse_batchnorm_with_conv"))
    options->enable(Algorithms::FuseBatchNormWithConv);
  if (arser.get<bool>("--fuse_add_to_fullyconnected_bias"))
    options->enable(Algorithms::FuseAddToFullyConnectedBias);
  if (arser.get<bool>("--fuse_add_with_conv"))
    options->enable(Algorithms::FuseAddWithConv);
  if (arser.get<bool>("--fuse_add_with_fully_connected"))
    options->enable(Algorithms::FuseAddWithFullyConnected);
  if (arser.get<bool>("--fuse_add_with_tconv"))
    options->enable(Algorithms::FuseAddWithTConv);
  if (arser.get<bool>("--fuse_batchnorm_with_dwconv"))
    options->enable(Algorithms::FuseBatchNormWithDwConv);
  if (arser.get<bool>("--fuse_batchnorm_with_tconv"))
    options->enable(Algorithms::FuseBatchNormWithTConv);
  if (arser.get<bool>("--fuse_mul_to_fullyconnected_weights"))
    options->enable(Algorithms::FuseMulToFullyConnectedWeights);
  if (arser.get<bool>("--fuse_slice_with_tconv"))
    options->enable(Algorithms::FuseSliceWithTConv);
  if (arser.get<bool>("--fuse_bcq"))
    options->enable(Algorithms::FuseBCQ);
  if (arser.get<bool>("--fuse_instnorm"))
    options->enable(Algorithms::FuseInstanceNorm);
  if (arser.get<bool>("--fuse_mean_with_mean"))
    options->enable(Algorithms::FuseMeanWithMean);
  if (arser.get<bool>("--fuse_mul_with_conv"))
    options->enable(Algorithms::FuseMulWithConv);
  if (arser.get<bool>("--fuse_mul_with_div"))
    options->enable(Algorithms::FuseMulWithDiv);
  if (arser.get<bool>("--make_batchnorm_gamma_positive"))
    options->enable(Algorithms::MakeBatchNormGammaPositive);
  if (arser.get<bool>("--fuse_preactivation_batchnorm"))
    options->enable(Algorithms::FusePreActivationBatchNorm);
  if (arser.get<bool>("--fuse_prelu"))
    options->enable(Algorithms::FusePRelu);
  if (arser.get<bool>("--fuse_gelu"))
    options->enable(Algorithms::FuseGelu);
  if (arser.get<bool>("--fuse_rsqrt"))
    options->enable(Algorithms::FuseRsqrt);
  if (arser.get<bool>("--fuse_transpose_with_mean"))
    options->enable(Algorithms::FuseTransposeWithMean);
  if (arser.get<bool>("--remove_duplicate_const"))
    options->enable(Algorithms::RemoveDuplicateConst);
  if (arser.get<bool>("--remove_fakequant"))
    options->enable(Algorithms::RemoveFakeQuant);
  if (arser.get<bool>("--remove_gather_guard"))
    options->enable(Algorithms::RemoveGatherGuard);
  if (arser.get<bool>("--remove_qdq_for_mpo"))
    options->enable(Algorithms::RemoveQDQForMixedPrecisionOp);
  if (arser.get<bool>("--remove_quantdequant"))
    options->enable(Algorithms::RemoveQuantDequantSeq);
  if (arser.get<bool>("--remove_redundant_quantize"))
    options->enable(Algorithms::RemoveRedundantQuantize);
  if (arser.get<bool>("--remove_redundant_reshape"))
    options->enable(Algorithms::RemoveRedundantReshape);
  if (arser.get<bool>("--remove_redundant_transpose"))
    options->enable(Algorithms::RemoveRedundantTranspose);
  if (arser.get<bool>("--remove_unnecessary_add"))
    options->enable(Algorithms::RemoveUnnecessaryAdd);
  if (arser.get<bool>("--remove_unnecessary_reshape"))
    options->enable(Algorithms::RemoveUnnecessaryReshape);
  if (arser.get<bool>("--remove_unnecessary_slice"))
    options->enable(Algorithms::RemoveUnnecessarySlice);
  if (arser.get<bool>("--remove_unnecessary_strided_slice"))
    options->enable(Algorithms::RemoveUnnecessaryStridedSlice);
  if (arser.get<bool>("--remove_unnecessary_split"))
    options->enable(Algorithms::RemoveUnnecessarySplit);
  if (arser.get<bool>("--remove_unnecessary_transpose"))
    options->enable(Algorithms::RemoveUnnecessaryTranspose);
  if (arser.get<bool>("--replace_cw_mul_add_with_depthwise_conv"))
    options->enable(Algorithms::ReplaceMulAddWithDepthwiseConv);
  if (arser.get<bool>("--replace_sub_with_add"))
    options->enable(Algorithms::ReplaceSubWithAdd);
  if (arser.get<bool>("--replace_with_fc_gelu_fc"))
    options->enable(Algorithms::ReplaceWithFCGeluFC);
  if (arser.get<bool>("--resolve_customop_add"))
    options->enable(Algorithms::ResolveCustomOpAdd);
  if (arser.get<bool>("--resolve_customop_batchmatmul"))
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
  if (arser.get<bool>("--resolve_customop_matmul"))
    options->enable(Algorithms::ResolveCustomOpMatMul);
  if (arser.get<bool>("--resolve_customop_max_pool_with_argmax"))
    options->enable(Algorithms::ResolveCustomOpMaxPoolWithArgmax);
  if (arser.get<bool>("--resolve_customop_splitv"))
    options->enable(Algorithms::ResolveCustomOpSplitV);
  if (arser.get<bool>("--resolve_former_customop"))
    options->enable(Algorithms::ResolveFormerCustomOp);
  if (arser.get<bool>("--shuffle_weight_to_16x1float32"))
    options->enable(Algorithms::ShuffleWeightTo16x1Float32);
  if (arser.get<bool>("--replace_non_const_fc_with_batch_matmul"))
    options->enable(Algorithms::ReplaceNonConstFCWithBatchMatMul);
  if (arser.get<bool>("--substitute_pack_to_reshape"))
    options->enable(Algorithms::SubstitutePackToReshape);
  if (arser.get<bool>("--substitute_padv2_to_pad"))
    options->enable(Algorithms::SubstitutePadV2ToPad);
  if (arser.get<bool>("--substitute_splitv_to_split"))
    options->enable(Algorithms::SubstituteSplitVToSplit);
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
  if (arser.get<bool>("--transform_sqrt_div_to_rsqrt_mul"))
    options->enable(Algorithms::TransformSqrtDivToRsqrtMul);
  if (arser.get<bool>("--common_subexpression_elimination"))
    options->enable(Algorithms::CommonSubExpressionElimination);
  if (arser.get<bool>("--decompose_hardswish"))
    options->enable(Algorithms::DecomposeHardSwishPass);
  if (arser.get<bool>("--decompose_softmax"))
    options->enable(Algorithms::DecomposeSoftmaxPass);
  if (arser.get<bool>("--expand_broadcast_const"))
    options->enable(Algorithms::ExpandBroadcastConst);
  if (arser.get<bool>("--unroll_unidirseqlstm"))
    options->enable(Algorithms::UnrollUnidirSeqLSTM);
  if (arser.get<bool>("--compress_weights_huffman"))
    options->enable(Algorithms::CompressWeightsHuffman);

  // NOTE Experimental options; these will be removed someday
  //      Add experimental options here
  // NOTE XpSepActFromTransposeConv is enabled for default
  //      exp_disable_sep_act_transposeconv is to turn it off
  //      which will leave TransposeConv with fused activation
  if (!arser.get<bool>("--exp_disable_sep_transposeconv_actfunc"))
    options->enable(Algorithms::XpSepActFromTransposeConv);

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

  bool dynamic_batch_to_single_batch = false;
  if (arser.get<bool>("--dynamic_batch_to_single_batch"))
  {
    dynamic_batch_to_single_batch = true;
  }

  // Import from input Circle file
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(input_path);
  if (module.get() == nullptr)
    return EXIT_FAILURE;

  // Convert dynamic batch to single batch
  // Why here? It has to be done before 'optimize', because most optimization
  // passes are written based on static shapes
  if (dynamic_batch_to_single_batch)
  {
    luci::dynamic_batch_to_single_batch(module.get());

    if (!luci::validate_shape(module.get()))
    {
      if (settings->get(luci::UserSettings::Key::DisableValidation))
        std::cerr
          << "WARNING: Invalid shape detected after converting dynamic batch to single batch"
          << std::endl;
      else
      {
        std::cerr << "ERROR: Invalid shape detected after converting dynamic batch to single batch"
                  << std::endl;
        return 255;
      }
    }
  }

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
