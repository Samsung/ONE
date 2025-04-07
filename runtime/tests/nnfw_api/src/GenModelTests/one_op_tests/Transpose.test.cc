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
#include <numeric>

TEST_F(GenModelTest, OneOp_Transpose_PermsToConst)
{
  CircleGen cgen;
  std::vector<int32_t> perms_data{2, 0, 1, 3};
  uint32_t perms_buf = cgen.addBuffer(perms_data);
  int perms = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, perms_buf});
  int in = cgen.addTensor({{2, 3, 4, 5}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 3, 4, 5}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
    {{0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
      18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
      36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
      54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
      72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
      90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
      108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119}},
    {{0,  1,  2,  3,  4,  20,  21,  22,  23,  24,  40, 41, 42, 43, 44, 60,  61,  62,  63,  64,
      80, 81, 82, 83, 84, 100, 101, 102, 103, 104, 5,  6,  7,  8,  9,  25,  26,  27,  28,  29,
      45, 46, 47, 48, 49, 65,  66,  67,  68,  69,  85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
      10, 11, 12, 13, 14, 30,  31,  32,  33,  34,  50, 51, 52, 53, 54, 70,  71,  72,  73,  74,
      90, 91, 92, 93, 94, 110, 111, 112, 113, 114, 15, 16, 17, 18, 19, 35,  36,  37,  38,  39,
      55, 56, 57, 58, 59, 75,  76,  77,  78,  79,  95, 96, 97, 98, 99, 115, 116, 117, 118, 119}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Transpose_PermsToVar)
{
  CircleGen cgen;
  int perms = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32});
  int in = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 3, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in, perms}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4, 5, 6})
                          .addInput<int32_t>({0, 2, 1, 3})
                          .addOutput<float>({1, 4, 2, 5, 3, 6}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Transpose_RegularTranspose)
{
  CircleGen cgen;
  int perms = cgen.addTensor({{0}, circle::TensorType::TensorType_INT32});
  int in = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 3, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in, perms}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4, 5, 6})
                          .addInput<int32_t>({})
                          .addOutput<float>({1, 4, 2, 5, 3, 6}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Transpose_6D)
{
  CircleGen cgen;

  std::vector<int32_t> in_shape{2, 3, 4, 5, 6, 2};
  std::vector<int32_t> perms_data{5, 3, 4, 0, 2, 1};
  std::vector<int32_t> out_shape{2, 5, 6, 2, 4, 3};

  std::vector<int32_t> in_data(
    std::accumulate(in_shape.begin(), in_shape.end(), 1, std::multiplies{}));
  std::iota(in_data.begin(), in_data.end(), 0);

  std::vector<int32_t> out_data(in_data.size());
  auto p = perms_data.data();
  auto i_ne = in_shape.data();
  auto o_ne = out_shape.data();

  // clang-format off
  for (int i0 = 0; i0 < i_ne[0]; ++i0)
    for (int i1 = 0; i1 < i_ne[1]; ++i1)
      for (int i2 = 0; i2 < i_ne[2]; ++i2)
        for (int i3 = 0; i3 < i_ne[3]; ++i3)
          for (int i4 = 0; i4 < i_ne[4]; ++i4)
            for (int i5 = 0; i5 < i_ne[5]; ++i5)
            {
              int i_idx = ((((i0 * i_ne[1] + i1) * i_ne[2] + i2) * i_ne[3] + i3) * i_ne[4] + i4) * i_ne[5] + i5;
              int o_i0 = p[0] == 0 ? i0 : p[0] == 1 ? i1 : p[0] == 2 ? i2 : p[0] == 3 ? i3 : p[0] == 4 ? i4 : i5;
              int o_i1 = p[1] == 0 ? i0 : p[1] == 1 ? i1 : p[1] == 2 ? i2 : p[1] == 3 ? i3 : p[1] == 4 ? i4 : i5;
              int o_i2 = p[2] == 0 ? i0 : p[2] == 1 ? i1 : p[2] == 2 ? i2 : p[2] == 3 ? i3 : p[2] == 4 ? i4 : i5;
              int o_i3 = p[3] == 0 ? i0 : p[3] == 1 ? i1 : p[3] == 2 ? i2 : p[3] == 3 ? i3 : p[3] == 4 ? i4 : i5;
              int o_i4 = p[4] == 0 ? i0 : p[4] == 1 ? i1 : p[4] == 2 ? i2 : p[4] == 3 ? i3 : p[4] == 4 ? i4 : i5;
              int o_i5 = p[5] == 0 ? i0 : p[5] == 1 ? i1 : p[5] == 2 ? i2 : p[5] == 3 ? i3 : p[5] == 4 ? i4 : i5;
              int o_idx = ((((o_i0 * o_ne[1] + o_i1) * o_ne[2] + o_i2) * o_ne[3] + o_i3) * o_ne[4] + o_i4) * o_ne[5] + o_i5;
              out_data[o_idx] = in_data[i_idx];
            }
  // clang-format on

  int in = cgen.addTensor({in_shape, circle::TensorType::TensorType_INT32});
  uint32_t perms_buf = cgen.addBuffer(perms_data);
  int perms = cgen.addTensor({{6}, circle::TensorType::TensorType_INT32, perms_buf});
  int out = cgen.addTensor({out_shape, circle::TensorType::TensorType_INT32});

  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}.addInput<int32_t>(in_data).addOutput<int32_t>(out_data));

  _context->setBackends({"cpu"});
  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Transpose_InvalidPermsSize)
{
  CircleGen cgen;
  std::vector<int32_t> perms_data{0, 1, 2};
  uint32_t perms_buf = cgen.addBuffer(perms_data);
  int perms = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32, perms_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Transpose_InvalidPermsVal)
{
  CircleGen cgen;
  std::vector<int32_t> perms_data{-3, 3, 1, 2};
  uint32_t perms_buf = cgen.addBuffer(perms_data);
  int perms = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, perms_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Transpose_DuplicatedPermsVal)
{
  CircleGen cgen;
  std::vector<int32_t> perms_data{3, 3, 1, 2};
  uint32_t perms_buf = cgen.addBuffer(perms_data);
  int perms = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, perms_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Transpose_DifferentType)
{
  CircleGen cgen;
  std::vector<int32_t> perms_data{3, 2, 1, 0};
  uint32_t perms_buf = cgen.addBuffer(perms_data);
  int perms = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, perms_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorTranspose({{in, perms}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
