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

TEST_F(GenModelTest, OneOp_ResizeNearestNeighbor)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> size_data{3, 3};
  uint32_t size_buf = cgen.addBuffer(size_data);
  int size = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, size_buf});

  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorResizeNearestNeighbor({{in, size}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{3, 4, 6, 10, 9, 10, 12, 16}},
                      {{3, 4, 3, 4, 6, 10, 3, 4, 3, 4, 6, 10, 9, 10, 9, 10, 12, 16}}));
  _context->setBackends({"acl_cl"});

  SUCCEED();
}
