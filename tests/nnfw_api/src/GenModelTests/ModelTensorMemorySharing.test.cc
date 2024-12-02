/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <nnfw_internal.h>

#include "CircleGen.h"
#include "GenModelTest.h"

namespace
{
// Add node other than Reshape/ExpandDims/Squeeze.
// It is used for cases where Reshape input/output is not input/output on the whole model.
uint32_t addNotOptimizedNode(CircleGen &cgen, const CircleGen::OperatorParams &params)
{
  return cgen.addOperatorCos(params);
}
} // namespace

TEST_F(GenModelTest, reshape_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape_out}, {cos2_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, expand_dims_inference)
{
  CircleGen cgen;
  std::vector<int32_t> axes_data{0, 1};
  uint32_t axes_buf = cgen.addBuffer(axes_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int axes = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, axes_buf});
  int expand_dims_out = cgen.addTensor({{1, 1, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{1, 1, 2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorExpandDims({{cos1_out, axes}, {expand_dims_out}});
  addNotOptimizedNode(cgen, {{expand_dims_out}, {cos2_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, squeeze_inference)
{
  CircleGen cgen;
  const std::vector<int32_t> squeeze_dims{0, 2};
  int input = cgen.addTensor({{1, 2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{1, 2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int squeeze_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorSqueeze({{cos1_out}, {squeeze_out}}, squeeze_dims);
  addNotOptimizedNode(cgen, {{squeeze_out}, {cos2_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_const_input_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  std::vector<float> reshape_in_data{1, 2, 3, 4};
  uint32_t reshape_in_buf = cgen.addBuffer(reshape_in_data);
  int reshape_input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32, reshape_in_buf});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorReshape({{reshape_input, new_shape}, {reshape_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape_out}, {cos_out}});
  cgen.setInputsAndOutputs({}, {cos_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({}, {{0.54030231, -0.41614684, -0.9899925, -0.65364362}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_dynamic_output)
{
  CircleGen cgen;
  int cast_in = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32});
  int cast_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int reshape_out = cgen.addTensor({{}, circle::TensorType::TensorType_FLOAT32});
  int cast2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32});

  cgen.addOperatorCast({{cast_in}, {cast_out}}, circle::TensorType::TensorType_INT32,
                       circle::TensorType::TensorType_FLOAT32);
  cgen.addOperatorReshape({{cast_out, new_shape}, {reshape_out}});
  cgen.addOperatorCast({{reshape_out}, {cast2_out}}, circle::TensorType::TensorType_FLOAT32,
                       circle::TensorType::TensorType_INT32);
  cgen.setInputsAndOutputs({cast_in, new_shape}, {cast2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int32_t>({{1, 2, 3, 4}, {2, 2}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_input_used_in_many_places_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos3_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape_out}, {cos2_out}});
  addNotOptimizedNode(cgen, {{cos1_out}, {cos3_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out, cos3_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345},
                                       {0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_output_used_in_many_places_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos3_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape_out}, {cos2_out}});
  addNotOptimizedNode(cgen, {{reshape_out}, {cos3_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out, cos3_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345},
                                       {0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_reshape_chain_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape1_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int reshape2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape1_out}}, &new_shape_data);
  cgen.addOperatorReshape({{reshape1_out, new_shape}, {reshape2_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape2_out}, {cos2_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_reshape_reshape_chain_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape1_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int reshape2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int reshape3_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape1_out}}, &new_shape_data);
  cgen.addOperatorReshape({{reshape1_out, new_shape}, {reshape2_out}}, &new_shape_data);
  cgen.addOperatorReshape({{reshape2_out, new_shape}, {reshape3_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape3_out}, {cos2_out}});
  cgen.setInputsAndOutputs({input}, {cos2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_input_model_input_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int output = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorReshape({{input, new_shape}, {output}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_input_model_output_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int cos2_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape_out}}, &new_shape_data);
  addNotOptimizedNode(cgen, {{reshape_out}, {cos2_out}});
  cgen.setInputsAndOutputs({input}, {cos1_out, cos2_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.54030231, -0.41614684, -0.9899925, -0.65364362},
                                       {0.85755322, 0.91465333, 0.54869613, 0.79387345}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, reshape_output_model_output_inference)
{
  CircleGen cgen;
  std::vector<int32_t> new_shape_data{2, 2};
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int cos1_out = cgen.addTensor({{4}, circle::TensorType::TensorType_FLOAT32});
  int new_shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, new_shape_buf});
  int reshape_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  addNotOptimizedNode(cgen, {{input}, {cos1_out}});
  cgen.addOperatorReshape({{cos1_out, new_shape}, {reshape_out}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {reshape_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 3, 4}}, {{0.54030231, -0.41614684, -0.9899925, -0.65364362}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}
