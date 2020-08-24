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

TEST_F(GenModelTest, OneOp_AvgPool2D)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _test_data = std::make_unique<GenModelTestData>(cgen.finish());
  _test_data->addTestCase({{{1, 3, 2, 4}}, {{2.5}}});
  TestableBackends({"acl_cl", "acl_neon", "cpu"});
}
