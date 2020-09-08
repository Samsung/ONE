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

void compute_sym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                          float &nudged_min, float &nudged_max);

void compute_asym_scale_zp(float min, float max, float &scaling_factor, int64_t &zp,
                           float &nudged_min, float &nudged_max);

bool get_channel_dim_index(CircleConst *node, loco::TensorShape &dimension, int &channel_dim_index);

uint32_t cal_offset(loco::TensorShape &dimension, uint32_t *indices);

void propagate_concat_quantparam(luci::CircleConcatenation *concat);

} // namespace luci

#endif // __LUCI_QUANTIZATION_UTILS_H__
