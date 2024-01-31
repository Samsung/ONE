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

TEST_F(GenModelTest, OneOp_L2Normalization)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 3}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorL2Normalization({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{0, 3, 4, 0, 5, 12, 0, 8, 15, 0, 7, 24}},
                      {{0, 0.6, 0.8, 0, 0.38461539149284363, 0.92307698726654053, 0,
                        0.47058823704719543, 0.88235294818878174, 0, 0.28, 0.96}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}
