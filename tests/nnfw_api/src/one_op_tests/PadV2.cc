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

TEST_F(GenModelTest, OneOp_PadV2)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  std::vector<float> padding_value_data{3.0};
  uint32_t padding_value_buf = cgen.addBuffer(padding_value_data);
  int padding_value =
      cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, padding_value_buf});

  int out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorPadV2({{in, padding, padding_value}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _test_data = std::make_unique<GenModelTestData>(cgen.finish());
  _test_data->addTestCase({{{1, 2, 3, 4}}, {{3, 3, 3, 3, 3, 1, 2, 3, 3, 3, 4, 3, 3, 3, 3, 3}}});
}
