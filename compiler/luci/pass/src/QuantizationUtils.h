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

#ifndef __LUCI_QUANTIZATION_UTILS_H__
#define __LUCI_QUANTIZATION_UTILS_H__

#include <luci/IR/CircleNodes.h>
#include <loco/IR/TensorShape.h>

namespace luci
{

// Compute scale using given min/max for symmetric quantization (int8/int16)
void compute_sym_scale(float min, float max, float &scaling_factor, float &nudged_min,
                       float &nudged_max, loco::DataType out_type = loco::DataType::S16);

// Compute scale/zp using given min/max for asymmetric quantization (uint8)
void compute_asym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                           float &nudged_min, float &nudged_max);

// Asymmetric per-layer quantization of weights (const tensor) using given min/max values
// NOTE: in-place update of node data
void asymmetric_wquant_with_minmax_per_layer(CircleConst *node, float min, float max,
                                             float &scaling_factor, int64_t &zp, float &nudged_min,
                                             float &nudged_max);

// Symmetric per-layer quantization of weights (const tensor) using given min/max values
// NOTE: in-place update of node data
void symmetric_wquant_with_minmax_per_layer(CircleConst *node, float min, float max,
                                            float &scaling_factor, float &nudged_min,
                                            float &nudged_max);

// Helper function to get channel dimension
// TODO Embed this function into iterate_per_channel
bool get_channel_dim_index(CircleConst *node, loco::TensorShape &dimension,
                           int32_t &channel_dim_index);

// Calculate offset of the given indices in dimension
uint32_t cal_offset(loco::TensorShape &dimension, uint32_t *indices);

// Backward propagation of concatenation qparam
void propagate_concat_quantparam(luci::CircleConcatenation *concat);

// Backward propagation of pad_v2 qparam
void propagate_pad_v2_quantparam(luci::CirclePadV2 *pad_v2);

// Return true if the node is quantized
bool is_quantized(const CircleNode *node);

// Return true if the node is fp32
bool is_fp32(const CircleNode *node);

enum ActivationQType
{
  MinMax,             // Quantize using recorded min/max
  PreDefinedLogistic, // Quantize using pre-defined values
  PreDefinedTanh,     // Quantize using pre-defined values
  PreDefinedSoftmax,  // Quantize using pre-defined values
  IntScale,           // Round scale to a positive integer
};

ActivationQType activation_qtype(const CircleNode *node);

// Create qparam with pre-defined values for speical operators
std::unique_ptr<CircleQuantParam> make_predefined_qparam(CircleNode *node, loco::DataType dtype);
std::unique_ptr<CircleQuantParam> make_predefined_qparam(ActivationQType qtype,
                                                         loco::DataType dtype,
                                                         CircleQuantParam *old_quant_param);

// Update node's scale to a positive integer (for special Ops e.g., Floor, Ceil)
void set_int_scale(luci::CircleNode *node);

// Quantize const tensor using its min/max values
void quant_const(luci::CircleConst *node, loco::DataType quant_type);

// Check that a node is quantized without significant loss of precision;
// Emits warnings to log with WARN
void warn_accuracy_with_range(luci::CircleNode *n);

} // namespace luci

#endif // __LUCI_QUANTIZATION_UTILS_H__
