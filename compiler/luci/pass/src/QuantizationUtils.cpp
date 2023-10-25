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

#include "QuantizationUtils.h"

#include <luci/Log.h>

#include <iostream>
#include <cmath>
#include <limits>

namespace luci
{

bool is_quantized(const CircleNode *node)
{
  return node->quantparam() != nullptr &&
         (node->dtype() == loco::DataType::U8 ||  // activation, weight (uint8 quant)
          node->dtype() == loco::DataType::S16 || // activation, weight (int16 quant)
          node->dtype() == loco::DataType::S32 || // bias (uint8 quant)
          node->dtype() == loco::DataType::S64);  // bias (int16 quant)
}

bool is_fp32(const CircleNode *node) { return node->dtype() == loco::DataType::FLOAT32; }

uint8_t fp32_to_uint8_cast(float f)
{
  assert(std::numeric_limits<uint8_t>::min() <= f);
  assert(f <= std::numeric_limits<uint8_t>::max());
  return static_cast<uint8_t>(f);
}

void asymmetric_wquant_with_minmax_per_layer(CircleConst *node, float min, float max,
                                             float &scaling_factor, int64_t &zp, float &nudged_min,
                                             float &nudged_max)
{
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  compute_asym_scale_zp(min, max, scaling_factor, zp, nudged_min, nudged_max);
  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    // clipping
    auto data = node->at<loco::DataType::FLOAT32>(i);
    data = data < nudged_min ? nudged_min : data;
    data = data > nudged_max ? nudged_max : data;
    quantized_values[i] =
      static_cast<int32_t>(std::round((data - nudged_min) * scaling_factor_inv));
  }

  node->dtype(loco::DataType::U8);      // change the type of tensor
  node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::U8>(i) = std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void symmetric_wquant_with_minmax_per_layer(CircleConst *node, float min, float max,
                                            float &scaling_factor, float &nudged_min,
                                            float &nudged_max)
{
  const int32_t kMaxScale = std::numeric_limits<int16_t>::max();
  const int32_t kMinScale = -kMaxScale;

  uint32_t size = node->size<loco::DataType::FLOAT32>();
  compute_sym_scale(min, max, scaling_factor, nudged_min, nudged_max);
  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    // clipping
    auto data = node->at<loco::DataType::FLOAT32>(i);
    data = data < nudged_min ? nudged_min : data;
    data = data > nudged_max ? nudged_max : data;
    quantized_values[i] = static_cast<int32_t>(std::round(data * scaling_factor_inv));
  }

  node->dtype(loco::DataType::S16);      // change the type of tensor
  node->size<loco::DataType::S16>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    node->at<loco::DataType::S16>(i) =
      std::min(kMaxScale, std::max(kMinScale, quantized_values[i]));
  }
}

void compute_sym_scale(float min, float max, float &scaling_factor, float &nudged_min,
                       float &nudged_max, loco::DataType out_type)
{
  assert(min <= max);
  assert(out_type == loco::DataType::S8 || out_type == loco::DataType::S16);

  const int32_t kMaxScale = (out_type == loco::DataType::S16) ? std::numeric_limits<int16_t>::max()
                                                              : std::numeric_limits<int8_t>::max();
  const int32_t kMinScale = -kMaxScale;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);
  double scale_factor_from_min_side{0};
  double scale_factor_from_max_side{0};

  if ((qmin_double * rmin) > 0)
    scale_factor_from_min_side = rmin / qmin_double;

  if ((qmax_double * rmax) > 0)
    scale_factor_from_max_side = rmax / qmax_double;

  scaling_factor = scale_factor_from_min_side > scale_factor_from_max_side
                     ? scale_factor_from_min_side
                     : scale_factor_from_max_side;

  // protect scale from being very low to avoid overflow/underflow
  const float kMinScalingFactor = (out_type == loco::DataType::S16) ? 1e-8 : 1e-5;
  scaling_factor = std::max(scaling_factor, kMinScalingFactor);

  nudged_min = static_cast<float>(qmin_double * scaling_factor);
  nudged_max = static_cast<float>(qmax_double * scaling_factor);
}

void compute_asym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                           float &nudged_min, float &nudged_max)
{
  LOGGER(l);

  assert(min <= max);
  const int32_t kMinScale = 0;
  const int32_t kMaxScale = 255;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const double rmin = std::fmin(0, min);
  const double rmax = std::fmax(0, max);

  double scale = (rmax - rmin) / (qmax_double - qmin_double);
  double zero_point_double = 0;
  uint8_t nudged_zero_point = 0;
  if (scale == 0)
  {
    WARN(l) << "The minimum and maximum values are the same." << std::endl;
    if (min >= 0 && max >= 0)
      zero_point_double = kMinScale;
    else
      zero_point_double = kMaxScale;
  }
  else
    zero_point_double = qmin_double - rmin / scale;
  if (min >= 0)
  {
    assert(min >= 0 && max >= 0);
    nudged_zero_point = kMinScale;
    scale = max / (qmax_double - qmin_double);
    if (min > 0 && max > 0)
      WARN(l) << "The minimum and maximum values are all positive." << std::endl;
  }
  else if (max < 0)
  {
    assert(min < 0 && max < 0);
    nudged_zero_point = kMaxScale;
    scale = -min / (qmax_double - qmin_double);
    WARN(l) << "The minimum and maximum values are all negative." << std::endl;
  }
  else
  {
    assert(min < 0 && max >= 0);
    nudged_zero_point = fp32_to_uint8_cast(std::round(zero_point_double));
  }

  // protect scale from being very low due to overflow
  if (scale < 1e-5)
  {
    scale = 1e-5;
    nudged_zero_point = fp32_to_uint8_cast(std::round(qmin_double - rmin / scale));
  }

  nudged_min = static_cast<float>((qmin_double - nudged_zero_point) * scale);
  nudged_max = static_cast<float>((qmax_double - nudged_zero_point) * scale);

  scaling_factor = scale;
  zp = nudged_zero_point;
}

bool get_channel_dim_index(CircleConst *node, loco::TensorShape &dimension,
                           int32_t &channel_dim_index)
{
  auto succs = loco::succs(node);

  // opcode is initialized to CIRCLEINPUT, because
  // CIRCLEINPUT should never be the successor of any node
  // (this is checked w/ the assert in the loop body)
  luci::CircleOpcode opcode = luci::CircleOpcode::CIRCLEINPUT;
  for (auto out : succs)
  {
    const auto circle_node = static_cast<CircleNode *>(out);
    assert(circle_node->opcode() != luci::CircleOpcode::CIRCLEINPUT);

    if (opcode == luci::CircleOpcode::CIRCLEINPUT)
    {
      opcode = circle_node->opcode();
    }
    else
    {
      // Node is used by multiple layers with different opcodes
      // We do not care such cases
      if (opcode != circle_node->opcode())
        return false;
    }
  }

  for (auto out : succs)
  {
    auto conv = dynamic_cast<CircleConv2D *>(out);
    auto dw_conv = dynamic_cast<CircleDepthwiseConv2D *>(out);
    auto tw_conv = dynamic_cast<CircleTransposeConv *>(out);
    auto fc = dynamic_cast<CircleFullyConnected *>(out);

    // Refer to https://github.com/Samsung/ONE/pull/2448.
    if ((conv != nullptr && conv->filter() == node) ||
        (tw_conv != nullptr && tw_conv->filter() == node)) // OHWI
    {
      assert(node->rank() == 4);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(node->dim(1).value());
      dimension.dim(2).set(node->dim(2).value());
      dimension.dim(3).set(node->dim(3).value());
      channel_dim_index = 0; // Set channel_dim_index based on "O"
      return true;
    }
    else if (dw_conv != nullptr && dw_conv->filter() == node) // IHWC
    {
      assert(node->rank() == 4);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(node->dim(1).value());
      dimension.dim(2).set(node->dim(2).value());
      dimension.dim(3).set(node->dim(3).value());
      channel_dim_index = 3; // Set channel_dim_index based on "C"
      return true;
    }
    else if (fc != nullptr && fc->weights() == node) // OI
    {
      assert(node->rank() == 2);
      dimension.dim(0).set(node->dim(0).value());
      dimension.dim(1).set(1); // Set FC layer like CONV
      dimension.dim(2).set(1);
      dimension.dim(3).set(node->dim(1).value());
      channel_dim_index = 0; // Set channel_dim_index based on "O"
      return true;
    }
    else
    {
      // node does not support channle-wise quantization
      assert(false);
    }
  }

  return false;
}

uint32_t cal_offset(loco::TensorShape &dimension, uint32_t *indices)
{
  return indices[0] * dimension.dim(1).value() * dimension.dim(2).value() *
           dimension.dim(3).value() +
         indices[1] * dimension.dim(2).value() * dimension.dim(3).value() +
         indices[2] * dimension.dim(3).value() + indices[3];
}

// Activation (ofm) qtype is determined in different ways.
// 1. Pre-defined values: Some Ops have pre-defined qparams (ex: LOGISTIC, TANH)
// 2. Integer scale: Output of some Ops should be integers (ex: FLOOR, CEIL)
// 3. Activation qtype of input: Some Ops propagate qparam from input to output (ex: QUANTIZE,
// TRANSPOSE, etc. See PropagateQParamForwardPass.cpp for more details).
ActivationQType activation_qtype(const CircleNode *node)
{
  auto fused_act_node = dynamic_cast<const CircleNodeMixin<CircleNodeTrait::FusedActFunc> *>(node);
  if (fused_act_node && fused_act_node->fusedActivationFunction() == FusedActFunc::TANH)
    return ActivationQType::PreDefinedTanh;

#define RETURN_INPUT_ACTIVATION_QTYPE(CLASS, INPUT)         \
  {                                                         \
    auto n = loco::must_cast<const CLASS *>(node);          \
    auto input = loco::must_cast<CircleNode *>(n->INPUT()); \
    return activation_qtype(input);                         \
  }

  switch (node->opcode())
  {
    case CircleOpcode::LOGISTIC:
      return ActivationQType::PreDefinedLogistic;
    case CircleOpcode::TANH:
      return ActivationQType::PreDefinedTanh;
    case CircleOpcode::SOFTMAX:
      return ActivationQType::PreDefinedSoftmax;
    case CircleOpcode::FLOOR:
    case CircleOpcode::FLOOR_DIV:
    case CircleOpcode::FLOOR_MOD:
    case CircleOpcode::CEIL:
      return ActivationQType::IntScale;
    case CircleOpcode::GATHER:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleGather, params);
    case CircleOpcode::RESHAPE:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleReshape, tensor);
    case CircleOpcode::TRANSPOSE:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleTranspose, a);
    case CircleOpcode::STRIDED_SLICE:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleStridedSlice, input);
    case CircleOpcode::SPLIT:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleSplit, input);
    case CircleOpcode::CIRCLESPLITOUT:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleSplitOut, input);
    case CircleOpcode::SPLIT_V:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleSplitV, input);
    case CircleOpcode::CIRCLESPLITVOUT:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleSplitVOut, input);
    case CircleOpcode::UNPACK:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleUnpack, value);
    case CircleOpcode::CIRCLEUNPACKOUT:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleUnpackOut, input);
    case CircleOpcode::QUANTIZE:
      RETURN_INPUT_ACTIVATION_QTYPE(CircleQuantize, input);
    default:
      break;
  }

#undef RETURN_INPUT_ACTIVATION_QTYPE

  return ActivationQType::MinMax;
}

std::unique_ptr<CircleQuantParam> make_predefined_qparam(ActivationQType qtype,
                                                         loco::DataType dtype,
                                                         CircleQuantParam *old_quant_param)
{
  auto qparam = std::make_unique<CircleQuantParam>();

  auto set_qparam = [&qparam, old_quant_param](float scale, int64_t zp) {
    qparam->scale.emplace_back(scale);
    qparam->zerop.emplace_back(zp);
    qparam->min = old_quant_param->min;
    qparam->max = old_quant_param->max;
  };

  switch (qtype)
  {
    case ActivationQType::PreDefinedLogistic:
      if (dtype == loco::DataType::U8)
        set_qparam(1.0f / 256.0f, 0);
      else
      {
        assert(dtype == loco::DataType::S16);
        set_qparam(1.0f / 32768.0f, 0);
      }
      break;
    case ActivationQType::PreDefinedTanh:
      if (dtype == loco::DataType::U8)
        set_qparam(2.0f / 256.0f, 128);
      else
      {
        assert(dtype == loco::DataType::S16);
        set_qparam(1.0f / 32768.0f, 0);
      }
      break;
    case ActivationQType::PreDefinedSoftmax:
      if (dtype == loco::DataType::U8)
        set_qparam(1.0f / 255.0f, 0);
      else
      {
        assert(dtype == loco::DataType::S16);
        set_qparam(1.0f / 32767.0f, 0);
      }
      break;
    default:
      throw std::runtime_error("Unsupported opcode with pre-defined qparam");
  }
  return qparam;
}

// For nodes with integer output, we use integer scale
void set_int_scale(luci::CircleNode *node)
{
  assert(node); // FIX_CALLER_UNLESS

  auto qparam = node->quantparam();
  assert(qparam);                    // FIX_CALLER_UNLESS
  assert(qparam->scale.size() == 1); // FIX_CALLER_UNLESS

  auto fp_scale = qparam->scale[0];
  qparam->scale[0] = fp_scale < 1 ? 1.0f : std::round(fp_scale);
}

void quant_const(luci::CircleConst *node, loco::DataType quant_type)
{
  assert(node->dtype() == loco::DataType::FLOAT32);

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  for (uint32_t i = 0; i < node->size<loco::DataType::FLOAT32>(); i++)
  {
    auto data = node->at<loco::DataType::FLOAT32>(i);
    min = data < min ? data : min;
    max = data > max ? data : max;
  }

  float scaling_factor{0.0};
  int64_t zp{0};
  float nudged_min{0.0};
  float nudged_max{0.0};

  switch (quant_type)
  {
    case loco::DataType::U8:
      asymmetric_wquant_with_minmax_per_layer(node, min, max, scaling_factor, zp, nudged_min,
                                              nudged_max);
      break;
    case loco::DataType::S16:
      symmetric_wquant_with_minmax_per_layer(node, min, max, scaling_factor, nudged_min,
                                             nudged_max);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
  }

  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale.push_back(scaling_factor);
  quantparam->zerop.push_back(zp);
  // Copy min and max values if it exists
  if (node->quantparam())
  {
    quantparam->min = node->quantparam()->min;
    quantparam->max = node->quantparam()->max;
  }
  node->quantparam(std::move(quantparam));
}

namespace
{

// TODO move this to a more global helper file
int nbits(loco::DataType dt) noexcept
{
  switch (dt)
  {
    case loco::DataType::S8:
    case loco::DataType::U8:
      return 8;
    case loco::DataType::S16:
    case loco::DataType::U16:
    case loco::DataType::FLOAT16:
      return 16;
    case loco::DataType::S32:
    case loco::DataType::U32:
    case loco::DataType::FLOAT32:
      return 32;
    case loco::DataType::S64:
      return 64;
    default:
      return 64; // a safe large default
  }
}

// TODO Check if the metric is valid
// Returns true if [min,max] is poorly representable
bool range_check(float min, float max, loco::DataType dtype)
{
  float thresh = 1.5f;
  return log2f(max) - log2f(min) > nbits(dtype) * thresh;
}

bool warn_scale_zp(float scale, int64_t zp, luci::CircleNode *n)
{
  float min, max;
  // estimate min/max
  switch (n->dtype())
  {
    case loco::DataType::U8:
      min = scale * (0 - zp);
      max = scale * (255 - zp);
      break;
    case loco::DataType::S16:
      min = scale * (-32767);
      max = scale * (32767);
      break;
    default:
      return false;
  }
  return range_check(min, max, n->dtype());
}

} // namespace

void warn_accuracy_with_range(luci::CircleNode *n)
{
  LOGGER(l);
  auto qp = n->quantparam();
  auto k = qp->zerop.size();
  for (uint32_t i = 0; i < k; i++)
  {
    if (warn_scale_zp(qp->scale[i], qp->zerop[i], n))
      WARN(l) << "Quantization of " << i << "-th channel of " << n->name()
              << "'s quantization may cause accuracy issues" << std::endl;
    ;
  }
}

} // namespace luci
