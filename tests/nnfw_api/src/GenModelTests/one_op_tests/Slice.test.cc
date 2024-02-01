/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

struct SliceVariationParam
{
  std::vector<int32_t> input_shape;
  std::vector<int32_t> begins;
  std::vector<int32_t> sizes;
  TestCaseData tcd;

  circle::TensorType input_type = circle::TensorType::TensorType_FLOAT32;
  float scale = 0.0f;
  int64_t zero_point = 0;
  circle::TensorType begins_type = circle::TensorType::TensorType_INT32;
};

class SliceVariation : public GenModelTest,
                       public ::testing::WithParamInterface<SliceVariationParam>
{
};

INSTANTIATE_TEST_SUITE_P(
  GenModelTest, SliceVariation,
  ::testing::Values(
    SliceVariationParam{
      {2, 2, 3, 1},
      {0, 1, 1, 0},
      {1, 1, 2, 1},
      uniformTCD<float>({{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}}, {{12, 13}})},
    SliceVariationParam{
      {2, 2, 3, 1},
      {0, 1, 1, 0},
      {1, 1, 2, 1},
      uniformTCD<uint8_t>({{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}}, {{12, 13}}),
      circle::TensorType::TensorType_UINT8,
      1,
      0},
    SliceVariationParam{
      {2, 2, 3, 1},
      {0, 1, 1, 0},
      {1, 1, 2, 1},
      uniformTCD<float>({{1, 2, 3, 11, 12, 13, 21, 22, 23, 31, 32, 33}}, {{12, 13}}),
      circle::TensorType::TensorType_FLOAT32,
      0,
      0,
      circle::TensorType::TensorType_INT64}));

TEST_P(SliceVariation, Test)
{
  auto &param = GetParam();

  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.input_type}, param.scale, param.zero_point);
  int out = cgen.addTensor({param.sizes, param.input_type}, param.scale, param.zero_point);
  if (param.begins_type == circle::TensorType::TensorType_INT32)
  {
    uint32_t begins_buf = cgen.addBuffer(param.begins);
    int rank = param.begins.size();
    int begins = cgen.addTensor({{rank}, param.begins_type, begins_buf});

    uint32_t sizes_buf = cgen.addBuffer(param.sizes);
    int sizes = cgen.addTensor({{rank}, param.begins_type, sizes_buf});

    cgen.addOperatorSlice({{in, begins, sizes}, {out}});
  }
  else if (param.begins_type == circle::TensorType::TensorType_INT64)
  {
    std::vector<int64_t> begins_64(param.begins.size());
    std::vector<int64_t> sizes_64(param.sizes.size());
    for (int i = 0; i < param.begins.size(); i++)
    {
      begins_64[i] = param.begins[i];
      sizes_64[i] = param.sizes[i];
    }

    uint32_t begins_buf = cgen.addBuffer(begins_64);
    int rank = param.begins.size();
    int begins = cgen.addTensor({{rank}, param.begins_type, begins_buf});

    uint32_t sizes_buf = cgen.addBuffer(sizes_64);
    int sizes = cgen.addTensor({{rank}, param.begins_type, sizes_buf});

    cgen.addOperatorSlice({{in, begins, sizes}, {out}});
  }
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);

  // acl don't support int64 yet
  if (param.begins_type == circle::TensorType::TensorType_INT64)
  {
    _context->setBackends({"cpu"});
  }
  else
  {
    _context->setBackends({"cpu", "acl_cl", "acl_neon"});
  }

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Slice_Type)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<float> begins_data = {0, 0, 1, 0};
  uint32_t begins_buf = cgen.addBuffer(begins_data);
  int begins = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, begins_buf});
  std::vector<float> sizes_data = {1, 2, 1, 1};
  uint32_t sizes_buf = cgen.addBuffer(sizes_data);
  int sizes = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, sizes_buf});
  int out = cgen.addTensor({{1, 2, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSlice({{in, begins, sizes}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_P(SliceVariation, neg_DiffType)
{
  auto &param = GetParam();

  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.input_type}, param.scale, param.zero_point);
  int out = cgen.addTensor({param.sizes, param.input_type}, param.scale, param.zero_point);
  if (param.begins_type == circle::TensorType::TensorType_INT32)
  {
    uint32_t begins_buf = cgen.addBuffer(param.begins);
    std::vector<int64_t> sizes_64(param.sizes.size());
    for (int i = 0; i < param.begins.size(); i++)
    {
      sizes_64[i] = param.sizes[i];
    }

    int rank = param.begins.size();
    int begins = cgen.addTensor({{rank}, param.begins_type, begins_buf});

    uint32_t sizes_buf = cgen.addBuffer(sizes_64);
    int sizes = cgen.addTensor({{rank}, circle::TensorType::TensorType_INT64, sizes_buf});

    cgen.addOperatorSlice({{in, begins, sizes}, {out}});
  }
  else if (param.begins_type == circle::TensorType::TensorType_INT64)
  {
    std::vector<int64_t> begins_64(param.begins.size());
    for (int i = 0; i < param.begins.size(); i++)
    {
      begins_64[i] = param.begins[i];
    }

    uint32_t begins_buf = cgen.addBuffer(begins_64);
    int rank = param.begins.size();
    int begins = cgen.addTensor({{rank}, param.begins_type, begins_buf});

    uint32_t sizes_buf = cgen.addBuffer(param.sizes);
    int sizes = cgen.addTensor({{rank}, circle::TensorType::TensorType_INT32, sizes_buf});

    cgen.addOperatorSlice({{in, begins, sizes}, {out}});
  }
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
