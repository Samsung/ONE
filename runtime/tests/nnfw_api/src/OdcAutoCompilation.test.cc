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
#include <gmock/gmock.h>

#include <nnfw_internal.h>

#include "common.h"
#include "fixtures.h"
#include "CircleGen.h"
#include "GenModelTest.h"
#include "NNPackages.h"

#include <chrono>
using ::testing::FloatNear;
using ::testing::Matcher;

Matcher<std::vector<float>> FloatArrayNear(const std::vector<float> &values, float max_abs_error)
{
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float v : values)
  {
    matchers.emplace_back(FloatNear(v, max_abs_error));
  }
  return ElementsAreArray(matchers);
}

const std::string model_name = "conv2d";

std::vector<std::vector<float>> input_tensors = {
  {1, 2, 3, 4, 5.5, 6.7, 1, 2, 3, 4, 5.5, 6.7, 1, 2, 3, 4, 5.5, 6.7},
  {7.1, 8, 9, 9.4, 1.2, 12.7, 1, 2, 3, 4, 5.5, 6.7, 7.1, 8, 9, 5.2, 3.2, 2.7},
  {6.3, 2, 3.2, 4.7, 5.5, 3.7, 1, 2, 3.5, 4.2, 5.5, 6.7, 1.7, 2.2, 3.5, 4.5, 5.5, 6.7},
  {8.4, 2.2, 3.7, 4.4, 5.5, 6.7, 1.2, 2.6, 3.3, 4.7, 5.5, 6.7, 1.3, 2, 3.2, 4.4, 5.5, 6.7},
  {1.2, 2.6, 3.3, 4, 5.5, 6.7, 1, 2, 3.4, 4, 5.5, 6.7, 1, 2.6, 3, 4, 5.5, 6.7}};

// Test for running a model with auto compilation
TEST(TestOdcAutoCompilation, AutoCompilation_test)
{
  EXPECT_TRUE(input_tensors.size());

  auto model_path = NNPackages::get().getModelAbsolutePath(model_name.c_str());

  // setup session and load model
  nnfw_session *session = nullptr;
  nnfw_create_session(&session);

  nnfw_load_model_from_file(session, (model_path + ".circle").c_str());
  nnfw_set_available_backends(session, "cpu");
  nnfw_prepare(session);

  // Delete minmax file
  nnfw_odc_delete_minmax_file(session);

  std::string compile_model_extension = "tvn";

  // Delete previuos quantized and compiled model
  std::string quantized_model_name = model_path + std::string(".q.circle");
  std::string compiled_model_name = model_path + std::string(".") + compile_model_extension;
  std::remove(quantized_model_name.c_str());
  std::remove(compiled_model_name.c_str());

  // setup ODC parameters
  nnfw_set_quantized_model_path(session, quantized_model_name.c_str());
  nnfw_set_quantization_type(session, NNFW_QUANTIZE_TYPE::NNFW_QUANTIZE_TYPE_U8_ASYM);

  nnfw_set_codegen_model_path(session, compiled_model_name.c_str());

  const int RUNS_COUNT_FOR_QUANTIZATION = input_tensors.size();
  nnfw_set_odc_param_minmax_records_count(session, RUNS_COUNT_FOR_QUANTIZATION);

  std::vector<std::vector<float>> float_model_output_tensors;
  std::vector<std::vector<float>> quantized_model_output_tensors;

  // Run FLOAT MODEL
  // prepare input and output data and run model
  for (int idx = 0; idx < RUNS_COUNT_FOR_QUANTIZATION; idx++)
  {

    // prepare input
    nnfw_tensorinfo ti;

    nnfw_input_tensorinfo(session, 0, &ti);
    nnfw_set_input(session, 0, ti.dtype, input_tensors[idx].data(),
                   sizeof(float) * input_tensors[idx].size());

    // prepare output
    nnfw_output_tensorinfo(session, 0, &ti);
    uint32_t output_elements = 1;
    for (int32_t i = 0; i < ti.rank; ++i)
      output_elements *= ti.dims[i];

    std::vector<float> output;
    output.resize(output_elements);
    nnfw_set_output(session, 0, ti.dtype, output.data(), sizeof(float) * output_elements);

    // run model
    NNFW_STATUS status =
      nnfw_run_with_auto_compilation(session, (compile_model_extension + "-gen").c_str(),
                                     NNFW_CODEGEN_PREF::NNFW_CODEGEN_PREF_DEFAULT);
    EXPECT_TRUE(status == NNFW_STATUS_NO_ERROR);

    float_model_output_tensors.push_back(output);
  }

  // Run COMPILED or QUANTIZED MODEL
  for (int idx = 0; idx < RUNS_COUNT_FOR_QUANTIZATION; idx++)
  {

    // prepare input
    nnfw_tensorinfo ti;

    nnfw_input_tensorinfo(session, 0, &ti);
    nnfw_set_input(session, 0, ti.dtype, input_tensors[idx].data(),
                   sizeof(float) * input_tensors[idx].size());

    // prepare output
    nnfw_output_tensorinfo(session, 0, &ti);
    uint32_t output_elements = 1;
    for (int32_t i = 0; i < ti.rank; ++i)
      output_elements *= ti.dims[i];

    std::vector<float> output;
    output.resize(output_elements);
    nnfw_set_output(session, 0, ti.dtype, output.data(), sizeof(float) * output_elements);

    // run quantized model
    NNFW_STATUS status =
      nnfw_run_with_auto_compilation(session, (compile_model_extension + "-gen").c_str(),
                                     NNFW_CODEGEN_PREF::NNFW_CODEGEN_PREF_DEFAULT);
    EXPECT_TRUE(status == NNFW_STATUS_NO_ERROR);

    quantized_model_output_tensors.push_back(output);
  }

  // results comparison
  for (size_t idx = 0; idx < quantized_model_output_tensors.size(); idx++)
  {
    EXPECT_THAT(float_model_output_tensors[idx],
                FloatArrayNear(quantized_model_output_tensors[idx], 0.1f));
  }

  nnfw_close_session(session);
  SUCCEED();
}

// Neg test for auto compilation
TEST(TestOdcAutoCompilation, neg_autoCompilation_no_export_path)
{

  EXPECT_TRUE(input_tensors.size());

  auto model_path = NNPackages::get().getModelAbsolutePath(model_name.c_str());

  // setup session and load model
  nnfw_session *session = nullptr;
  nnfw_create_session(&session);

  nnfw_load_model_from_file(session, (model_path + ".circle").c_str());
  nnfw_set_available_backends(session, "cpu");
  nnfw_prepare(session);

  // Delete minmax file
  nnfw_odc_delete_minmax_file(session);

  const int RUNS_COUNT_FOR_QUANTIZATION = 1;
  nnfw_set_odc_param_minmax_records_count(session, RUNS_COUNT_FOR_QUANTIZATION);

  // Run FLOAT MODEL
  // prepare input
  nnfw_tensorinfo ti;

  nnfw_input_tensorinfo(session, 0, &ti);
  nnfw_set_input(session, 0, ti.dtype, input_tensors[0].data(),
                 sizeof(float) * input_tensors[0].size());

  // prepare output
  nnfw_output_tensorinfo(session, 0, &ti);
  uint32_t output_elements = 1;
  for (int32_t i = 0; i < ti.rank; ++i)
    output_elements *= ti.dims[i];

  std::vector<float> output;
  output.resize(output_elements);
  nnfw_set_output(session, 0, ti.dtype, output.data(), sizeof(float) * output_elements);

  // run model
  NNFW_STATUS status =
    nnfw_run_with_auto_compilation(session, "", NNFW_CODEGEN_PREF::NNFW_CODEGEN_PREF_DEFAULT);
  ASSERT_EQ(status, NNFW_STATUS_INVALID_STATE);

  nnfw_close_session(session);
  SUCCEED();
}
