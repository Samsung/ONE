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

CircleBuffer genSimpleMeanModel()
{
  CircleGen cgen;
  uint32_t axis_buf = cgen.addBuffer(std::vector<int32_t>{1, 2});
  int in = cgen.addTensor({{1, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorMean({{in, axis}, {out}}, true);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

TEST_F(GenModelTest, OneOp_Mean)
{
  auto model = genSimpleMeanModel();
  _context = std::make_unique<GenModelTestContext>(std::move(model));
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4, 5, 6, 7, 8, 9}}, {{5}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

CircleBuffer genWrongMeanModel()
{
  CircleGen cgen;
  uint32_t axis_buf = cgen.addBuffer(std::vector<int32_t>{1, 2});
  int in = cgen.addTensor({{1, 3, 3, 1}, circle::TensorType::TensorType_BOOL});
  int axis = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_BOOL});
  cgen.addOperatorMean({{in, axis}, {out}}, true);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

TEST_F(GenModelTest, neg_OneOp_Mean)
{
  auto model = genWrongMeanModel();
  _context = std::make_unique<GenModelTestContext>(std::move(model));
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4, 5, 6, 7, 8, 9}}, {{5}}));
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}
