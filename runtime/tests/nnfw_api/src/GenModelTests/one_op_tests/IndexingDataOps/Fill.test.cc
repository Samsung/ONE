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

struct FillVariationParam
{
  TestCaseData tcd;
  const uint8_t *value_data = nullptr;
  circle::TensorType data_type = circle::TensorType::TensorType_FLOAT32;
};

class FillVariation : public GenModelTest, public ::testing::WithParamInterface<FillVariationParam>
{
};

// value is constant
TEST_P(FillVariation, Test)
{
  auto &param = GetParam();

  CircleGen cgen;

  size_t value_size =
    (param.data_type == circle::TensorType::TensorType_INT64) ? sizeof(int64_t) : sizeof(int32_t);
  uint32_t value_buf = cgen.addBuffer(param.value_data, value_size);

  int dims = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int value = cgen.addTensor({{1}, param.data_type, value_buf});
  int out = cgen.addTensor({{2, 3}, param.data_type});
  cgen.addOperatorFill({{dims, value}, {out}});
  cgen.setInputsAndOutputs({dims}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);
  _context->setBackends({"cpu"});

  SUCCEED();
}

const int32_t test_int32 = 13;
const int64_t test_int64 = 1052;
const float test_float = 5.2;

// Test with different value type
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, FillVariation,
  ::testing::Values(
    // float value
    FillVariationParam{
      TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<float>({5.2, 5.2, 5.2, 5.2, 5.2, 5.2}),
      reinterpret_cast<const uint8_t *>(&test_float)},
    // int32 value
    FillVariationParam{
      TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int32_t>({13, 13, 13, 13, 13, 13}),
      reinterpret_cast<const uint8_t *>(&test_int32), circle::TensorType::TensorType_INT32},
    // uint8 value
    FillVariationParam{
      TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int64_t>({1052, 1052, 1052, 1052, 1052,
                                                                   1052}),
      reinterpret_cast<const uint8_t *>(&test_int64), circle::TensorType::TensorType_INT64}));

TEST_F(GenModelTest, OneOp_Fill_Int64_Shape)
{
  CircleGen cgen;
  std::vector<float> value_data{1.3};
  uint32_t value_buf = cgen.addBuffer(value_data);

  int dims = cgen.addTensor({{2}, circle::TensorType::TensorType_INT64});
  int value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, value_buf});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFill({{dims, value}, {out}});
  cgen.setInputsAndOutputs({dims}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int64_t>({2, 3}).addOutput<float>({1.3, 1.3, 1.3, 1.3, 1.3, 1.3}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Fill_Int32_oneoperand)
{
  CircleGen cgen;

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorFill({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int32_t>({13, 13, 13, 13, 13, 13}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Fill_Int64_oneoperand)
{
  CircleGen cgen;

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT64});
  cgen.addOperatorFill({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int64_t>({13, 13, 13, 13, 13, 13}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Fill_Float32_oneoperand)
{
  CircleGen cgen;

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFill({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<float>({1.3, 1.3, 1.3, 1.3, 1.3, 1.3}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
