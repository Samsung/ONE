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
  add_switch(arser, "--fuse_mul_with_fullyconnected",
             "This will fuse Mul operator with a preceding FullyConnected operator.");
  add_switch(arser, "--fuse_mul_with_rmsnorm",
             "This will fuse Mul operator with a preceding RmsNorm operator.");
  add_switch(arser, "--fuse_rmsnorm", "This will fuse operators to RmsNorm operator");
  add_switch(arser, "--fuse_rope", "This will fuse operators to rope operator");
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
  add_switch(arser, "--remove_unnecessary_cast",
             "This will remove unnecessary cast with the same input and output type.");
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
  add_switch(arser, "--substitute_expand_dims_to_reshape",
             "This will convert ExpandDims with constant axis to Reshape");
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
  add_switch(arser, "--convert_nchw_to_nhwc",
             "This will convert NCHW operators to NHWC under the assumption that "
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

  // clang-format off
  std::map<std::string /* option string */, Algorithms /* option enum */> option_str_to_enum;
  option_str_to_enum["fold_add_v2"] = Algorithms::FoldAddV2;
  option_str_to_enum["fold_cast"] = Algorithms::FoldCast;
  option_str_to_enum["fold_densify"] = Algorithms::FoldDensify;
  option_str_to_enum["fold_dequantize"] = Algorithms::FoldDequantize;
  option_str_to_enum["fold_dwconv"] = Algorithms::FoldDepthwiseConv2D;
  option_str_to_enum["fold_fully_connected"] = Algorithms::FoldFullyConnected;
  option_str_to_enum["fold_gather"] = Algorithms::FoldGather;
  option_str_to_enum["fold_mul"] = Algorithms::FoldMul;
  option_str_to_enum["fold_reshape"] = Algorithms::FoldReshape;
  option_str_to_enum["fold_shape"] = Algorithms::FoldShape;
  option_str_to_enum["fold_sparse_to_dense"] = Algorithms::FoldSparseToDense;
  option_str_to_enum["fold_squeeze"] = Algorithms::FoldSqueeze;
  option_str_to_enum["forward_reshape_to_unaryop"] = Algorithms::ForwardReshapeToUnaryOp;
  option_str_to_enum["forward_transpose_op"] = Algorithms::ForwardTransposeOp;
  option_str_to_enum["fuse_activation_function"] = Algorithms::FuseActivationFunction;
  option_str_to_enum["fuse_horizontal_fc_layers"] = Algorithms::FuseHorizontalFullyConnected;
  option_str_to_enum["fuse_batchnorm_with_conv"] = Algorithms::FuseBatchNormWithConv;
  option_str_to_enum["fuse_add_to_fullyconnected_bias"] = Algorithms::FuseAddToFullyConnectedBias;
  option_str_to_enum["fuse_add_with_conv"] = Algorithms::FuseAddWithConv;
  option_str_to_enum["fuse_add_with_fully_connected"] = Algorithms::FuseAddWithFullyConnected;
  option_str_to_enum["fuse_add_with_tconv"] = Algorithms::FuseAddWithTConv;
  option_str_to_enum["fuse_batchnorm_with_dwconv"] = Algorithms::FuseBatchNormWithDwConv;
  option_str_to_enum["fuse_batchnorm_with_tconv"] = Algorithms::FuseBatchNormWithTConv;
  option_str_to_enum["fuse_mul_to_fullyconnected_weights"] = Algorithms::FuseMulToFullyConnectedWeights;
  option_str_to_enum["fuse_slice_with_tconv"] = Algorithms::FuseSliceWithTConv;
  option_str_to_enum["fuse_bcq"] = Algorithms::FuseBCQ;
  option_str_to_enum["fuse_instnorm"] = Algorithms::FuseInstanceNorm;
  option_str_to_enum["fuse_mean_with_mean"] = Algorithms::FuseMeanWithMean;
  option_str_to_enum["fuse_mul_with_conv"] = Algorithms::FuseMulWithConv;
  option_str_to_enum["fuse_mul_with_div"] = Algorithms::FuseMulWithDiv;
  option_str_to_enum["fuse_mul_with_fullyconnected"] = Algorithms::FuseMulWithFullyConnected;
  option_str_to_enum["fuse_mul_with_rmsnorm"] = Algorithms::FuseMulWithRmsNorm;
  option_str_to_enum["make_batchnorm_gamma_positive"] = Algorithms::MakeBatchNormGammaPositive;
  option_str_to_enum["fuse_preactivation_batchnorm"] = Algorithms::FusePreActivationBatchNorm;
  option_str_to_enum["fuse_prelu"] = Algorithms::FusePRelu;
  option_str_to_enum["fuse_gelu"] = Algorithms::FuseGelu;
  option_str_to_enum["fuse_rmsnorm"] = Algorithms::FuseRmsNorm;
  option_str_to_enum["fuse_rope"] = Algorithms::FuseRoPE;
  option_str_to_enum["fuse_rsqrt"] = Algorithms::FuseRsqrt;
  option_str_to_enum["fuse_transpose_with_mean"] = Algorithms::FuseTransposeWithMean;
  option_str_to_enum["remove_duplicate_const"] = Algorithms::RemoveDuplicateConst;
  option_str_to_enum["remove_fakequant"] = Algorithms::RemoveFakeQuant;
  option_str_to_enum["remove_gather_guard"] = Algorithms::RemoveGatherGuard;
  option_str_to_enum["remove_qdq_for_mpo"] = Algorithms::RemoveQDQForMixedPrecisionOp;
  option_str_to_enum["remove_quantdequant"] = Algorithms::RemoveQuantDequantSeq;
  option_str_to_enum["remove_redundant_quantize"] = Algorithms::RemoveRedundantQuantize;
  option_str_to_enum["remove_redundant_reshape"] = Algorithms::RemoveRedundantReshape;
  option_str_to_enum["remove_redundant_transpose"] = Algorithms::RemoveRedundantTranspose;
  option_str_to_enum["remove_unnecessary_add"] = Algorithms::RemoveUnnecessaryAdd;
  option_str_to_enum["remove_unnecessary_cast"] = Algorithms::RemoveUnnecessaryCast;
  option_str_to_enum["remove_unnecessary_reshape"] = Algorithms::RemoveUnnecessaryReshape;
  option_str_to_enum["remove_unnecessary_slice"] = Algorithms::RemoveUnnecessarySlice;
  option_str_to_enum["remove_unnecessary_strided_slice"] = Algorithms::RemoveUnnecessaryStridedSlice;
  option_str_to_enum["remove_unnecessary_split"] = Algorithms::RemoveUnnecessarySplit;
  option_str_to_enum["remove_unnecessary_transpose"] = Algorithms::RemoveUnnecessaryTranspose;
  option_str_to_enum["replace_cw_mul_add_with_depthwise_conv"] = Algorithms::ReplaceMulAddWithDepthwiseConv;
  option_str_to_enum["replace_sub_with_add"] = Algorithms::ReplaceSubWithAdd;
  option_str_to_enum["replace_with_fc_gelu_fc"] = Algorithms::ReplaceWithFCGeluFC;
  option_str_to_enum["resolve_customop_add"] = Algorithms::ResolveCustomOpAdd;
  option_str_to_enum["resolve_customop_batchmatmul"] = Algorithms::ResolveCustomOpBatchMatMul;
  option_str_to_enum["resolve_customop_matmul"] = Algorithms::ResolveCustomOpMatMul;
  option_str_to_enum["resolve_customop_max_pool_with_argmax"] = Algorithms::ResolveCustomOpMaxPoolWithArgmax;
  option_str_to_enum["resolve_customop_splitv"] = Algorithms::ResolveCustomOpSplitV;
  option_str_to_enum["resolve_former_customop"] = Algorithms::ResolveFormerCustomOp;
  option_str_to_enum["shuffle_weight_to_16x1float32"] = Algorithms::ShuffleWeightTo16x1Float32;
  option_str_to_enum["replace_non_const_fc_with_batch_matmul"] = Algorithms::ReplaceNonConstFCWithBatchMatMul;
  option_str_to_enum["substitute_expand_dims_to_reshape"] = Algorithms::SubstituteExpandDimsToReshape;
  option_str_to_enum["substitute_pack_to_reshape"] = Algorithms::SubstitutePackToReshape;
  option_str_to_enum["substitute_padv2_to_pad"] = Algorithms::SubstitutePadV2ToPad;
  option_str_to_enum["substitute_splitv_to_split"] = Algorithms::SubstituteSplitVToSplit;
  option_str_to_enum["substitute_squeeze_to_reshape"] = Algorithms::SubstituteSqueezeToReshape;
  option_str_to_enum["substitute_strided_slice_to_reshape"] = Algorithms::SubstituteStridedSliceToReshape;
  option_str_to_enum["substitute_transpose_to_reshape"] = Algorithms::SubstituteTransposeToReshape;
  option_str_to_enum["transform_min_max_to_relu6"] = Algorithms::TransformMinMaxToRelu6Pass;
  option_str_to_enum["transform_min_relu_to_relu6"] = Algorithms::TransformMinReluToRelu6Pass;
  option_str_to_enum["transform_sqrt_div_to_rsqrt_mul"] = Algorithms::TransformSqrtDivToRsqrtMul;
  option_str_to_enum["common_subexpression_elimination"] = Algorithms::CommonSubExpressionElimination;
  option_str_to_enum["decompose_hardswish"] = Algorithms::DecomposeHardSwishPass;
  option_str_to_enum["decompose_softmax"] = Algorithms::DecomposeSoftmaxPass;
  option_str_to_enum["expand_broadcast_const"] = Algorithms::ExpandBroadcastConst;
  option_str_to_enum["unroll_unidirseqlstm"] = Algorithms::UnrollUnidirSeqLSTM;
  // clang-format on

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }
  for (auto const &x : option_str_to_enum)
  {
    if (arser.get<bool>("--" + x.first))
      options->enable(x.second);
  }

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
