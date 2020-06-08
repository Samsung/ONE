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

#include "OpPrinter.h"
#include "Read.h"

#include <stdex/Memory.h>

#include <flatbuffers/flexbuffers.h>

using stdex::make_unique;

namespace circledump
{

// TODO move to some header
std::ostream &operator<<(std::ostream &os, const std::vector<int32_t> &vect);

// TODO Re-arrange in alphabetical order

class AddPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_AddOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class ArgMaxPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_ArgMaxOptions())
    {
      os << "    ";
      os << "OutputType(" << EnumNameTensorType(params->output_type()) << ") ";
      os << std::endl;
    }
  }
};

class BatchMatMulPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_BatchMatMulOptions())
    {
      os << "    ";
      os << std::boolalpha;
      os << "adjoint_lhs(" << params->adjoint_lhs() << ") ";
      os << "adjoint_rhs(" << params->adjoint_rhs() << ") ";
      os << std::endl;
    }
  }
};

class CastPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto cast_params = op->builtin_options_as_CastOptions())
    {
      os << "    ";
      os << "in_data_type(" << circle::EnumNameTensorType(cast_params->in_data_type()) << ") ";
      os << "out_data_type(" << circle::EnumNameTensorType(cast_params->out_data_type()) << ") ";
      os << std::endl;
    }
  }
};

class Conv2DPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto conv_params = op->builtin_options_as_Conv2DOptions())
    {
      os << "    ";
      os << "Padding(" << conv_params->padding() << ") ";
      os << "Stride.W(" << conv_params->stride_w() << ") ";
      os << "Stride.H(" << conv_params->stride_h() << ") ";
      os << "Dilation.W(" << conv_params->dilation_w_factor() << ") ";
      os << "Dilation.H(" << conv_params->dilation_h_factor() << ")";
      os << "Activation("
         << EnumNameActivationFunctionType(conv_params->fused_activation_function()) << ")";
      os << std::endl;
    }
  }
};

class DepthToSpacePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *std_params = op->builtin_options_as_DepthToSpaceOptions())
    {
      os << "    ";
      os << "BlockSize(" << std_params->block_size() << ")";
      os << std::endl;
    }
  }
};

class DivPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_DivOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class Pool2DPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto pool_params = op->builtin_options_as_Pool2DOptions())
    {
      os << "    ";
      os << "Padding(" << pool_params->padding() << ") ";
      os << "Stride.W(" << pool_params->stride_w() << ") ";
      os << "Stride.H(" << pool_params->stride_h() << ") ";
      os << "Filter.W(" << pool_params->filter_width() << ") ";
      os << "Filter.H(" << pool_params->filter_height() << ") ";
      os << "Activation("
         << EnumNameActivationFunctionType(pool_params->fused_activation_function()) << ")";
      os << std::endl;
    }
  }
};

class ConcatenationPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *concatenation_params = op->builtin_options_as_ConcatenationOptions())
    {
      os << "    ";
      os << "Activation("
         << EnumNameActivationFunctionType(concatenation_params->fused_activation_function())
         << ") ";
      os << "Axis(" << concatenation_params->axis() << ")";
      os << std::endl;
    }
  }
};

class ReducerPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto reducer_params = op->builtin_options_as_ReducerOptions())
    {
      os << "    ";
      os << "keep_dims(" << reducer_params->keep_dims() << ") ";
      os << std::endl;
    }
  }
};

class ReshapePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *reshape_params = op->builtin_options_as_ReshapeOptions())
    {
      auto new_shape = circleread::as_index_vector(reshape_params->new_shape());
      os << "    ";
      os << "NewShape(" << new_shape << ")";
      os << std::endl;
    }
  }
};

class ResizeBilinearPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *resize_params = op->builtin_options_as_ResizeBilinearOptions())
    {
      os << "    ";
      os << std::boolalpha;
      os << "align_corners(" << resize_params->align_corners() << ")";
      os << "half_pixel_centers(" << resize_params->half_pixel_centers() << ")";
      os << std::endl;
    }
  }
};

class ResizeNearestNeighborPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *resize_params = op->builtin_options_as_ResizeNearestNeighborOptions())
    {
      os << "    ";
      os << std::boolalpha;
      os << "align_corners(" << resize_params->align_corners() << ")";
      os << std::endl;
    }
  }
};

class DepthwiseConv2DPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto conv_params = op->builtin_options_as_DepthwiseConv2DOptions())
    {
      os << "    ";
      os << "Padding(" << conv_params->padding() << ") ";
      os << "Stride.W(" << conv_params->stride_w() << ") ";
      os << "Stride.H(" << conv_params->stride_h() << ") ";
      os << "DepthMultiplier(" << conv_params->depth_multiplier() << ") ";
      os << "Dilation.W(" << conv_params->dilation_w_factor() << ") ";
      os << "Dilation.H(" << conv_params->dilation_h_factor() << ")";
      os << "Activation("
         << EnumNameActivationFunctionType(conv_params->fused_activation_function()) << ") ";
      os << std::endl;
    }
  }
};

class FullyConnectedPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_FullyConnectedOptions())
    {
      os << "    ";
      os << "WeightFormat(" << EnumNameFullyConnectedOptionsWeightsFormat(params->weights_format())
         << ") ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";

      os << std::endl;
    }
  }
};

class GatherPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_GatherOptions())
    {
      os << "    ";
      os << "Axis(" << params->axis() << ") ";

      os << std::endl;
    }
  }
};

class IfPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_IfOptions())
    {
      os << "    ";
      os << "then_subgraph_index(" << params->then_subgraph_index() << ") ";
      os << "else_subgraph_index(" << params->else_subgraph_index() << ") ";
      os << std::endl;
    }
  }
};

class L2NormPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_L2NormOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class LeakyReluPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_LeakyReluOptions())
    {
      os << "    ";
      os << "alpha(" << params->alpha() << ") ";
    }
  }
};

class LocalResponseNormalizationPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_LocalResponseNormalizationOptions())
    {
      os << "    ";
      os << "radius(" << params->radius() << ") ";
      os << "bias(" << params->bias() << ") ";
      os << "alpha(" << params->alpha() << ") ";
      os << "beta(" << params->beta() << ") ";
      os << std::endl;
    }
  }
};

class MirrorPadPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_MirrorPadOptions())
    {
      os << "    ";
      os << "mode(" << EnumNameMirrorPadMode(params->mode()) << ") ";
      os << std::endl;
    }
  }
};

class MulPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_MulOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class OneHotPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_OneHotOptions())
    {
      os << "    ";
      os << "Axis(" << params->axis() << ") ";

      os << std::endl;
    }
  }
};

class PackPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_PackOptions())
    {
      os << "    ";
      os << "ValuesCount(" << params->values_count() << ") ";
      os << "Axis(" << params->axis() << ") ";
      os << std::endl;
    }
  }
};

class ShapePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_ShapeOptions())
    {
      os << "    ";
      os << "out_type(" << EnumNameTensorType(params->out_type()) << ") ";
      os << std::endl;
    }
  }
};

class SoftmaxPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *softmax_params = op->builtin_options_as_SoftmaxOptions())
    {
      os << "    ";
      os << "Beta(" << softmax_params->beta() << ")";
      os << std::endl;
    }
  }
};

class SpaceToDepthPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *std_params = op->builtin_options_as_SpaceToDepthOptions())
    {
      os << "    ";
      os << "BlockSize(" << std_params->block_size() << ")";
      os << std::endl;
    }
  }
};

class SplitPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_SplitOptions())
    {
      os << "    ";
      os << "num_splits(" << params->num_splits() << ") ";
      os << std::endl;
    }
  }
};

class SplitVPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_SplitVOptions())
    {
      os << "    ";
      os << "num_splits(" << params->num_splits() << ") ";
      os << std::endl;
    }
  }
};

class SqueezePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_SqueezeOptions())
    {
      os << "    ";
      os << "SqueezeDims(";
      for (int i = 0; i < params->squeeze_dims()->size(); ++i)
      {
        if (i != 0)
          os << ", ";
        os << params->squeeze_dims()->Get(i);
      }
      os << ")";
      os << std::endl;
    }
  }
};

class StridedSlicePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *strided_slice_params = op->builtin_options_as_StridedSliceOptions())
    {
      os << "    ";
      os << "begin_mask(" << strided_slice_params->begin_mask() << ") ";
      os << "end_mask(" << strided_slice_params->end_mask() << ") ";
      os << "ellipsis_mask(" << strided_slice_params->ellipsis_mask() << ") ";
      os << "new_axis_mask(" << strided_slice_params->new_axis_mask() << ") ";
      os << "shrink_axis_mask(" << strided_slice_params->shrink_axis_mask() << ") ";
      os << std::endl;
    }
  }
};

class SubPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_SubOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << std::endl;
    }
  }
};

class TransposeConvPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto conv_params = op->builtin_options_as_TransposeConvOptions())
    {
      os << "    ";
      os << "Padding(" << conv_params->padding() << ") ";
      os << "Stride.W(" << conv_params->stride_w() << ") ";
      os << "Stride.H(" << conv_params->stride_h() << ") ";
      os << std::endl;
    }
  }
};

class WhilePrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_WhileOptions())
    {
      os << "    ";
      os << "cond_subgraph_index(" << params->cond_subgraph_index() << ") ";
      os << "body_subgraph_index(" << params->body_subgraph_index() << ") ";
      os << std::endl;
    }
  }
};

class CustomOpPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (op->custom_options_format() != circle::CustomOptionsFormat::CustomOptionsFormat_FLEXBUFFERS)
    {
      os << "    ";
      os << "Unknown custom option format";
      return;
    }

    const flatbuffers::Vector<uint8_t> *option_buf = op->custom_options();

    if (option_buf == nullptr || option_buf->size() == 0)
    {
      os << "No attrs found." << std::endl;
      return;
    }

    // printing attrs
    // attrs of custom ops are encoded in flexbuffer format
    auto attr_map = flexbuffers::GetRoot(option_buf->data(), option_buf->size()).AsMap();

    os << "    ";
    auto keys = attr_map.Keys();
    for (int i = 0; i < keys.size(); i++)
    {
      auto key = keys[i].ToString();
      os << key << "(" << attr_map[key].ToString() << ") ";
    }

    // Note: attr in "Shape" type does not seem to be converted by circle_convert.
    // When the converted circle file (with custom op) is opened with hexa editory,
    // attrs names can be found but attr name in "Shape" type is not found.

    os << std::endl;
  }
};

class BCQFullyConnectedPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_BCQFullyConnectedOptions())
    {
      os << "    ";
      os << "Activation(" << EnumNameActivationFunctionType(params->fused_activation_function())
         << ") ";
      os << "weights_hidden_size(" << params->weights_hidden_size() << ") ";
      os << std::endl;
    }
  }
};

class BCQGatherPrinter : public OpPrinter
{
public:
  void options(const circle::Operator *op, std::ostream &os) const override
  {
    if (auto *params = op->builtin_options_as_BCQGatherOptions())
    {
      os << "    ";
      os << "axis(" << params->axis() << ") ";
      os << "weights_hidden_size(" << params->input_hidden_size() << ") ";
      os << std::endl;
    }
  }
};

OpPrinterRegistry::OpPrinterRegistry()
{
  _op_map[circle::BuiltinOperator_ADD] = make_unique<AddPrinter>();
  _op_map[circle::BuiltinOperator_ARG_MAX] = make_unique<ArgMaxPrinter>();
  _op_map[circle::BuiltinOperator_AVERAGE_POOL_2D] = make_unique<Pool2DPrinter>();
  _op_map[circle::BuiltinOperator_BATCH_MATMUL] = make_unique<BatchMatMulPrinter>();
  _op_map[circle::BuiltinOperator_CAST] = make_unique<CastPrinter>();
  _op_map[circle::BuiltinOperator_CONCATENATION] = make_unique<ConcatenationPrinter>();
  _op_map[circle::BuiltinOperator_CONV_2D] = make_unique<Conv2DPrinter>();
  _op_map[circle::BuiltinOperator_DEPTH_TO_SPACE] = make_unique<DepthToSpacePrinter>();
  _op_map[circle::BuiltinOperator_DEPTHWISE_CONV_2D] = make_unique<DepthwiseConv2DPrinter>();
  _op_map[circle::BuiltinOperator_DIV] = make_unique<DivPrinter>();
  // There is no Option for FLOOR
  // There is no Option for FLOOR_MOD
  _op_map[circle::BuiltinOperator_FULLY_CONNECTED] = make_unique<FullyConnectedPrinter>();
  _op_map[circle::BuiltinOperator_GATHER] = make_unique<GatherPrinter>();
  _op_map[circle::BuiltinOperator_IF] = make_unique<IfPrinter>();
  _op_map[circle::BuiltinOperator_L2_NORMALIZATION] = make_unique<L2NormPrinter>();
  _op_map[circle::BuiltinOperator_L2_POOL_2D] = make_unique<Pool2DPrinter>();
  _op_map[circle::BuiltinOperator_LEAKY_RELU] = make_unique<LeakyReluPrinter>();
  _op_map[circle::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION] =
      make_unique<LocalResponseNormalizationPrinter>();
  // There is no Option for LOG
  // There is no Option for LOGISTIC
  _op_map[circle::BuiltinOperator_MAX_POOL_2D] = make_unique<Pool2DPrinter>();
  _op_map[circle::BuiltinOperator_MIRROR_PAD] = make_unique<MirrorPadPrinter>();
  _op_map[circle::BuiltinOperator_MUL] = make_unique<MulPrinter>();
  _op_map[circle::BuiltinOperator_ONE_HOT] = make_unique<OneHotPrinter>();
  _op_map[circle::BuiltinOperator_PACK] = make_unique<PackPrinter>();
  // There is no Option for PAD
  // There is no Option for PRELU
  // There is no Option for RELU
  // There is no Option for RELU6
  // There is no Option for RELU_N1_TO_1
  _op_map[circle::BuiltinOperator_REDUCE_ANY] = make_unique<ReducerPrinter>();
  _op_map[circle::BuiltinOperator_REDUCE_MAX] = make_unique<ReducerPrinter>();
  _op_map[circle::BuiltinOperator_REDUCE_MIN] = make_unique<ReducerPrinter>();
  _op_map[circle::BuiltinOperator_REDUCE_PROD] = make_unique<ReducerPrinter>();
  _op_map[circle::BuiltinOperator_RESHAPE] = make_unique<ReshapePrinter>();
  _op_map[circle::BuiltinOperator_RESIZE_BILINEAR] = make_unique<ResizeBilinearPrinter>();
  _op_map[circle::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR] =
      make_unique<ResizeNearestNeighborPrinter>();
  // There is no Option for SELECT
  _op_map[circle::BuiltinOperator_SHAPE] = make_unique<ShapePrinter>();
  // There is no Option for SIN
  // There is no Option for SLICE
  _op_map[circle::BuiltinOperator_SOFTMAX] = make_unique<SoftmaxPrinter>();
  _op_map[circle::BuiltinOperator_SPACE_TO_DEPTH] = make_unique<SpaceToDepthPrinter>();
  // There is no Option for SPACE_TO_BATCH_ND
  _op_map[circle::BuiltinOperator_SPLIT] = make_unique<SplitPrinter>();
  _op_map[circle::BuiltinOperator_SPLIT_V] = make_unique<SplitVPrinter>();
  _op_map[circle::BuiltinOperator_SQUEEZE] = make_unique<SqueezePrinter>();
  _op_map[circle::BuiltinOperator_STRIDED_SLICE] = make_unique<StridedSlicePrinter>();
  _op_map[circle::BuiltinOperator_SUB] = make_unique<SubPrinter>();
  _op_map[circle::BuiltinOperator_SUM] = make_unique<ReducerPrinter>();
  _op_map[circle::BuiltinOperator_TRANSPOSE_CONV] = make_unique<TransposeConvPrinter>();
  // There is no Option for TOPK_V2
  _op_map[circle::BuiltinOperator_WHILE] = make_unique<WhilePrinter>();
  _op_map[circle::BuiltinOperator_CUSTOM] = make_unique<CustomOpPrinter>();

  // Circle only
  _op_map[circle::BuiltinOperator_BCQ_FULLY_CONNECTED] = make_unique<BCQFullyConnectedPrinter>();
  _op_map[circle::BuiltinOperator_BCQ_GATHER] = make_unique<BCQGatherPrinter>();
}

} // namespace circledump
