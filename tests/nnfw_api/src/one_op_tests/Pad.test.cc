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

// Input shape: {1, 2, 2, 1}
// Padding: {0, 0, 1, 1, 1, 1, 0, 0}
// Output shape: {1, 4, 4, 1}
struct PadParam
{
  TestCaseData tcd;
  circle::TensorType data_type = circle::TensorType::TensorType_FLOAT32;
  float scale = 0.0f;
  int64_t zero_point = 0;
};

class PadVariation : public GenModelTest, public ::testing::WithParamInterface<PadParam>
{
};

// Test with different value type
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, PadVariation,
  ::testing::Values(
    // float value
    PadParam{uniformTCD<float>({{1, 2, 3, 4}}, {{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0}})},
    // uint8 value
    PadParam{
      uniformTCD<uint8_t>({{1, 2, 3, 4}}, {{8, 8, 8, 8, 8, 1, 2, 8, 8, 3, 4, 8, 8, 8, 8, 8}}),
      circle::TensorType::TensorType_UINT8, 1.0, 8},
    // int8 value
    PadParam{uniformTCD<int8_t>({{-2, -1, 1, 2}},
                                {{-5, -5, -5, -5, -5, -2, -1, -5, -5, 1, 2, -5, -5, -5, -5, -5}}),
             circle::TensorType::TensorType_INT8, 1.0, -5}));

TEST_P(PadVariation, Test)
{
  auto &param = GetParam();

  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, param.data_type}, param.scale, param.zero_point);
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, param.data_type}, param.scale, param.zero_point);

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_P(PadVariation, neg_InvalidPadRank)
{
  auto &param = GetParam();

  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, param.data_type}, param.scale, param.zero_point);
  std::vector<int32_t> padding_data{1, 1, 1, 1};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, param.data_type}, param.scale, param.zero_point);

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_P(PadVariation, neg_InvalidPadDim0)
{
  auto &param = GetParam();

  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, param.data_type}, param.scale, param.zero_point);
  std::vector<int32_t> padding_data{1, 1, 1, 1};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, param.data_type}, param.scale, param.zero_point);

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_P(PadVariation, neg_InvalidPadDim1)
{
  auto &param = GetParam();

  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, param.data_type}, param.scale, param.zero_point);
  std::vector<int32_t> padding_data{1, 1, 1, 1};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 1}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, param.data_type}, param.scale, param.zero_point);

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_P(PadVariation, neg_Type)
{
  auto &param = GetParam();

  const circle::TensorType output_type = ((param.data_type == circle::TensorType::TensorType_UINT8)
                                            ? circle::TensorType::TensorType_INT8
                                            : circle::TensorType::TensorType_UINT8);

  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, param.data_type}, param.scale, param.zero_point);
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, output_type}, 1.0, 0);

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Pad_QuantParam)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 1.0, 1);
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_UINT8}, 1.0, 3);

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
