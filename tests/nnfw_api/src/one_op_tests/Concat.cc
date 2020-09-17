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

TEST_F(GenModelTest, OneOp_Concat_ShareSubTensor)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int shared_subtensor = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int concat_out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int pad_out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {shared_subtensor}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorConcatenation({{rhs, shared_subtensor}, {concat_out}}, 3,
                                circle::ActivationFunctionType_NONE);
  cgen.addOperatorPad({{shared_subtensor, padding}, {pad_out}});
  cgen.setInputsAndOutputs({lhs, rhs}, {pad_out, concat_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
      {{1, 3, 2, 4}, {5, 4, 7, 4}},
      {{0, 0, 0, 0, 0, 6, 7, 0, 0, 9, 8, 0, 0, 0, 0, 0}, {5, 6, 4, 7, 7, 9, 4, 8}}));
  _context->setBackends({"acl_cl", "acl_neon"});

  SUCCEED();
}
