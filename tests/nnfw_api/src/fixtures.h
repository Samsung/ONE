/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_API_TEST_FIXTURES_H__
#define __NNFW_API_TEST_FIXTURES_H__

#include <array>
#include <gtest/gtest.h>
#include <nnfw.h>

#include "model_path.h"

inline uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    n *= ti->dims[i];
  }
  return n;
}

struct SessionObject
{
  nnfw_session *session = nullptr;
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<float>> outputs;
};

class ValidationTest : public ::testing::Test
{
protected:
  void SetUp() override {}
};

class ValidationTestSessionCreated : public ValidationTest
{
protected:
  void SetUp() override
  {
    ValidationTest::SetUp();
    ASSERT_EQ(nnfw_create_session(&_session), NNFW_STATUS_NO_ERROR);
  }

  void TearDown() override
  {
    ASSERT_EQ(nnfw_close_session(_session), NNFW_STATUS_NO_ERROR);
    ValidationTest::TearDown();
  }

protected:
  nnfw_session *_session = nullptr;
};

class ValidationTestOneOpModelLoaded : public ValidationTestSessionCreated
{
protected:
  void SetUp() override
  {
    ValidationTestSessionCreated::SetUp();
    ASSERT_EQ(nnfw_load_model_from_file(
                  _session, ModelPath::get().getModelAbsolutePath(MODEL_ONE_OP_IN_TFLITE).c_str()),
              NNFW_STATUS_NO_ERROR);
    ASSERT_NE(_session, nullptr);
  }

  void TearDown() override { ValidationTestSessionCreated::TearDown(); }
};

class ValidationTestFourOneOpModelSetInput : public ValidationTest
{
protected:
  static const uint32_t NUM_SESSIONS = 4;

  void SetUp() override
  {
    ValidationTest::SetUp();

    auto model_path = ModelPath::get().getModelAbsolutePath(MODEL_ONE_OP_IN_TFLITE);
    for (auto &obj : _objects)
    {
      ASSERT_EQ(nnfw_create_session(&obj.session), NNFW_STATUS_NO_ERROR);
      ASSERT_EQ(nnfw_load_model_from_file(obj.session, model_path.c_str()), NNFW_STATUS_NO_ERROR);
      ASSERT_EQ(nnfw_prepare(obj.session), NNFW_STATUS_NO_ERROR);

      obj.inputs.resize(1);
      nnfw_tensorinfo ti;
      ASSERT_EQ(nnfw_input_tensorinfo(obj.session, 0, &ti), NNFW_STATUS_NO_ERROR);
      uint64_t input_elements = num_elems(&ti);
      obj.inputs[0].resize(input_elements);
      ASSERT_EQ(nnfw_set_input(obj.session, 0, ti.dtype, obj.inputs[0].data(),
                               sizeof(float) * input_elements),
                NNFW_STATUS_NO_ERROR);

      obj.outputs.resize(1);
      nnfw_tensorinfo ti_output;
      ASSERT_EQ(nnfw_output_tensorinfo(obj.session, 0, &ti_output), NNFW_STATUS_NO_ERROR);
      uint64_t output_elements = num_elems(&ti_output);
      obj.outputs[0].resize(output_elements);
      ASSERT_EQ(nnfw_set_output(obj.session, 0, ti_output.dtype, obj.outputs[0].data(),
                                sizeof(float) * output_elements),
                NNFW_STATUS_NO_ERROR);
    }
  }

  void TearDown() override
  {
    for (auto &obj : _objects)
    {
      ASSERT_EQ(nnfw_close_session(obj.session), NNFW_STATUS_NO_ERROR);
    }
    ValidationTest::TearDown();
  }

protected:
  std::array<SessionObject, NUM_SESSIONS> _objects;
};

#endif // __NNFW_API_TEST_FIXTURES_H__
