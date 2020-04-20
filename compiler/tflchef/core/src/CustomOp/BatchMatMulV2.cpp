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

#include "BatchMatMulV2.h"

flatbuffers::Offset<void> BatchMatMulV2Chef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  return flatbuffers::Offset<void>();
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
BatchMatMulV2Chef::custom_value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.type() == "BatchMatMulV2");

  uint8_t adj_x = operation.batch_matmul_options().adjoint_lhs() ? 1 : 0;
  uint8_t adj_y = operation.batch_matmul_options().adjoint_rhs() ? 1 : 0;

  std::vector<uint8_t> custom_options_vec = {0x61,  0x64, 0x6a, 0x5f, 0x78, 0x0,  0x61,
                                             0x64,  0x6a, 0x5f, 0x79, 0x0,  0x54, 0x0,
                                             0x3,   0x3,  0x10, 0xb,  0x3,  0x1,  0x3,
                                             0x0,   // TensorType
                                             adj_x, // adj_x
                                             adj_y, // adj_y
                                             0x4,   0x68, 0x68, 0x6,  0x24, 0x1};
  //{97, 100, 106, 95, 120, 0, 97, 100, 106, 95, 121, 0, 84, 0, 3, 3, 16, 11, 3, 1, 3, 0, 0,
  // 0, 4, 104, 104, 6, 36, 1}

  auto circle_custom_options = fbb.CreateVector(custom_options_vec);
  return circle_custom_options;
}

std::unique_ptr<OpChef> BatchMatMulV2ChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new BatchMatMulV2Chef{operation}};
}
