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

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{2.5}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_AvgPool2D_Large)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 16, 32, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 16, 16, 16, 16,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({std::vector<float>(1024, 99)}, {{99, 99, 99, 99}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_3DInput)
{
  // 3D Tensors are not supported
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_2DInput)
{
  // 2D Tensors are not supported
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_InvalidPaddingType)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, static_cast<circle::Padding>(99), 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_InvalidFilterSize_1)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, -1, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_InvalidFilterSize_2)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 0,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_InvalidStrides_1)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 0, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_InvalidStrides_2)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 1, -100, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
