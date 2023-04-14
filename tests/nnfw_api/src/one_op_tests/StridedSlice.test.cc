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

TEST_F(GenModelTest, OneOp_StridedSlice_LastDim)
{
  CircleGen cgen;
  std::vector<int32_t> begin_data{0, 3};
  std::vector<int32_t> end_data{0, 6};
  std::vector<int32_t> strides_data{1, 1};
  uint32_t begin_buf = cgen.addBuffer(begin_data);
  uint32_t end_buf = cgen.addBuffer(end_data);
  uint32_t strides_buf = cgen.addBuffer(strides_data);
  int input = cgen.addTensor({{1, 6}, circle::TensorType::TensorType_FLOAT32});
  int begin = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, begin_buf});
  int end = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, end_buf});
  int strides = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, strides_buf});
  int out = cgen.addTensor({{1, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorStridedSlice({{input, begin, end, strides}, {out}}, 1, 1);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4, 5, 6}}, {{4, 5, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_StridedSlice_DifferentType)
{
  CircleGen cgen;
  std::vector<int32_t> begin_data{0, 3};
  std::vector<int32_t> end_data{0, 6};
  std::vector<int32_t> strides_data{1, 1};
  uint32_t begin_buf = cgen.addBuffer(begin_data);
  uint32_t end_buf = cgen.addBuffer(end_data);
  uint32_t strides_buf = cgen.addBuffer(strides_data);
  int input = cgen.addTensor({{1, 6}, circle::TensorType::TensorType_FLOAT32});
  int begin = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, begin_buf});
  int end = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, end_buf});
  int strides = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, strides_buf});
  int out = cgen.addTensor({{1, 3}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorStridedSlice({{input, begin, end, strides}, {out}}, 1, 1);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4, 5, 6}}, {{4, 5, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
