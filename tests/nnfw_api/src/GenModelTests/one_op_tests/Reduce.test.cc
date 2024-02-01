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

CircleBuffer genSimpleReduceModel(circle::BuiltinOperator op, bool keep_dims)
{
  CircleGen cgen;
  uint32_t axis_buf = cgen.addBuffer(std::vector<int32_t>{0, 1, 2, 3});
  int in = cgen.addTensor({{2, 1, 1, 3}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorReduce({{in, axis}, {out}}, op, keep_dims);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

TEST_F(GenModelTest, OneOp_ReduceMax)
{
  auto model = genSimpleReduceModel(circle::BuiltinOperator_REDUCE_MAX, false);
  _context = std::make_unique<GenModelTestContext>(std::move(model));
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4, 5, 6}}, {{6}}));
  _context->addTestCase(uniformTCD<float>({{100, 98, 55, 200, 3, 40}}, {{200}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

class ReduceMaxBadIndex : public GenModelTest,
                          public ::testing::WithParamInterface<std::vector<int>>
{
};

TEST_P(ReduceMaxBadIndex, neg_Test)
{
  CircleGen cgen;
  // Axis cannot be equal or bigger than input's rank - 4
  uint32_t axis_buf = cgen.addBuffer(GetParam());
  int in = cgen.addTensor({{2, 1, 1, 3}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorReduce({{in, axis}, {out}}, circle::BuiltinOperator_REDUCE_MAX, false);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailCompile();

  SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(GenModelTest, ReduceMaxBadIndex,
                         ::testing::Values(std::vector<int32_t>{0, 1, 2, 4},
                                           std::vector<int32_t>{0, -5, 2, 3},
                                           std::vector<int32_t>{-88, 1, 2, 3},
                                           std::vector<int32_t>{0, 1, 88, 3}));
