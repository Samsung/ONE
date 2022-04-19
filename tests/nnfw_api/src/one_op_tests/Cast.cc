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

#include <memory>

CircleGen genSimpleCastModel(circle::TensorType from_t, circle::TensorType to_t)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, from_t});
  int out = cgen.addTensor({{1, 2, 2, 1}, to_t});
  cgen.addOperatorCast({{in}, {out}}, from_t, to_t);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen;
}

TEST_F(GenModelTest, OneOp_Cast_Int32ToFloat32)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_INT32, circle::TensorType_FLOAT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({1, 2, 3, 4}).addOutput<float>({1, 2, 3, 4}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_Float32ToInt32)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_FLOAT32, circle::TensorType_INT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<float>({1, 2, 3, 4}).addOutput<int32_t>({1, 2, 3, 4}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_BoolToFloat32)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_BOOL, circle::TensorType_FLOAT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<bool>({true, false, true, true}).addOutput<float>({1, 0, 1, 1}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_BoolToUInt8)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_BOOL, circle::TensorType_UINT8);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<bool>({true, false, true, true})
                          .addOutput(std::vector<uint8_t>{1, 0, 1, 1}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_BoolToInt32)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_BOOL, circle::TensorType_INT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<bool>({true, false, true, true}).addOutput<int32_t>({1, 0, 1, 1}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_Uint8ToFloat32)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_UINT8, circle::TensorType_FLOAT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  // clang-format off
  _context->addTestCase(
    TestCaseData{}.addInput<uint8_t>({0, 100, 200, 255})
                  .addOutput<float>({0., 100., 200., 255.}));
  // clang-format on
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_Int64ToFloat32)
{
  CircleGen cgen = genSimpleCastModel(circle::TensorType_INT64, circle::TensorType_FLOAT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<int64_t>({-12345, 3, 100, 2147483648})
                          .addOutput<float>({-12345., 3., 100., 2147483648.}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Cast_AfterEqual)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int equal_out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorEqual({{lhs, rhs}, {equal_out}});
  cgen.addOperatorCast({{equal_out}, {out}}, circle::TensorType::TensorType_BOOL,
                       circle::TensorType::TensorType_FLOAT32);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}, {2, 3, 1, 4}}, {{0, 1, 0, 1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Cast_InvalidInputCount0)
{
  CircleGen cgen;
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorCast({{}, {out}}, circle::TensorType::TensorType_FLOAT32,
                       circle::TensorType::TensorType_INT32);
  cgen.setInputsAndOutputs({}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Cast_InvalidInputCount2)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1, 2, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorCast({{lhs, rhs}, {out}}, circle::TensorType::TensorType_INT32,
                       circle::TensorType::TensorType_FLOAT32);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Cast_InvalidOutputCount0)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorCast({{in}, {}}, circle::TensorType::TensorType_INT32,
                       circle::TensorType::TensorType_FLOAT32);
  cgen.setInputsAndOutputs({in}, {});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Cast_InvalidOutputCount2)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int out1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorCast({{in}, {out1, out2}}, circle::TensorType::TensorType_INT32,
                       circle::TensorType::TensorType_FLOAT32);
  cgen.setInputsAndOutputs({in}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
