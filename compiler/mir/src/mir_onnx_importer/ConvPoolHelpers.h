/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef MIR_ONNX_CONV_POOL_HELPERS_H
#define MIR_ONNX_CONV_POOL_HELPERS_H

#include "mir/Shape.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mir_onnx
{

void inferAutoPadding(const std::string &pad_type, const mir::Shape &input_shape,
                      const std::vector<std::int32_t> &dilations,
                      const std::vector<std::int32_t> &strides,
                      const std::vector<std::int32_t> &window_size,
                      std::vector<std::int32_t> &padding_before,
                      std::vector<std::int32_t> &padding_after);

std::vector<std::int32_t> fixPads(const mir::Shape &input_shape,
                                  const std::vector<std::int32_t> &pads,
                                  const std::vector<std::int32_t> &strides,
                                  const std::vector<std::int32_t> &dilation,
                                  const std::vector<std::int32_t> &kernel_shape);

} // namespace mir_onnx

#endif // MIR_ONNX_CONV_POOL_HELPERS_H
