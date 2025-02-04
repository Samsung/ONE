/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "GenModelTest.h"

struct DepthToSpaceVariationParam
{
  TestCaseData tcd;
  circle::TensorType type = circle::TensorType::TensorType_FLOAT32;
  float scale = 0.0f;
  int64_t zero_point = 0;
};

class DepthToSpaceVariation : public GenModelTest,
                              public ::testing::WithParamInterface<DepthToSpaceVariationParam>
{
};

// Input shape: {1, 1, 2, 4}
// Block size: 2
// Output shape: {1, 2, 4, 1}
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, DepthToSpaceVariation,
  ::testing::Values(
    // Float
    DepthToSpaceVariationParam{
      uniformTCD<float>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6, 3, 4, 7, 8}})},
    // Int32
    DepthToSpaceVariationParam{
      uniformTCD<int32_t>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6, 3, 4, 7, 8}}),
      circle::TensorType::TensorType_INT32},
    // Int64
    DepthToSpaceVariationParam{
      uniformTCD<int64_t>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6, 3, 4, 7, 8}}),
      circle::TensorType::TensorType_INT64},
    // Uint8
    DepthToSpaceVariationParam{
      uniformTCD<uint8_t>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6, 3, 4, 7, 8}}),
      circle::TensorType::TensorType_UINT8, 1.0f, -2},
    // Int8
    DepthToSpaceVariationParam{
      uniformTCD<int8_t>({{1, 2, 3, 4, 5, 6, 7, 8}}, {{1, 2, 5, 6, 3, 4, 7, 8}}),
      circle::TensorType::TensorType_INT8, 1.0f, -2}));

TEST_P(DepthToSpaceVariation, Test)
{
  auto &param = GetParam();

  CircleGen cgen;
  int in = cgen.addTensor({{1, 1, 2, 4}, param.type}, param.scale, param.zero_point);
  int out = cgen.addTensor({{1, 2, 4, 1}, param.type}, param.scale, param.zero_point);
  cgen.addOperatorDepthToSpace({{in}, {out}}, 2);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_P(DepthToSpaceVariation, neg_Blocksize)
{
  auto &param = GetParam();

  CircleGen cgen;
  int in = cgen.addTensor({{1, 1, 2, 4}, param.type}, param.scale, param.zero_point);
  int out = cgen.addTensor({{1, 2, 4, 1}, param.type}, param.scale, param.zero_point);
  cgen.addOperatorDepthToSpace({{in}, {out}}, -2);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
