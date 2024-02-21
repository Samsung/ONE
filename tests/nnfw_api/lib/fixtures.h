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
#include <nnfw_experimental.h>
#include <nnfw_internal.h>

#include "NNPackages.h"
#include "CircleGen.h"

#define NNFW_ENSURE_SUCCESS(EXPR) ASSERT_EQ((EXPR), NNFW_STATUS_NO_ERROR)

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

class RegressionTest : public ::testing::Test
{
protected:
  void SetUp() override {}
};

class ValidationTestSingleSession : public ValidationTest
{
protected:
  nnfw_session *_session = nullptr;
};

class ValidationTestSessionCreated : public ValidationTestSingleSession
{
protected:
  void SetUp() override
  {
    ValidationTestSingleSession::SetUp();
    ASSERT_EQ(nnfw_create_session(&_session), NNFW_STATUS_NO_ERROR);
    ASSERT_NE(_session, nullptr);
  }

  void TearDown() override
  {
    ASSERT_EQ(nnfw_close_session(_session), NNFW_STATUS_NO_ERROR);
    ValidationTestSingleSession::TearDown();
  }
};

inline CircleBuffer genAddModel()
{
  CircleGen cgen;
  std::vector<float> rhs_data{2};
  uint32_t rhs_buf = cgen.addBuffer(rhs_data);
  int lhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, 0, "X_input"});
  int rhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, rhs_buf, "y_var"});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, 0, "ADD_TOP"});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs}, {out});
  return cgen.finish();
}

template <int PackageNo> class ValidationTestModelLoaded : public ValidationTestSessionCreated
{
protected:
  void SetUp() override
  {
    ValidationTestSessionCreated::SetUp();
    if (PackageNo == NNPackages::ADD)
    {
      // NOTE the circle buffer must be kept until finishing the test, so keep it as a member
      _cbuf = genAddModel();
      NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(_session, _cbuf.buffer(), _cbuf.size()));
    }
    else
    {
      // TODO Eventually, downloaded model tests are removed.
      NNFW_ENSURE_SUCCESS(nnfw_load_model_from_file(
        _session, NNPackages::get().getModelAbsolutePath(PackageNo).c_str()));
    }
  }

  void TearDown() override { ValidationTestSessionCreated::TearDown(); }

private:
  CircleBuffer _cbuf; // Used only for models from buffer, unused for models from files
};

template <int PackageNo>
class ValidationTestSessionPrepared : public ValidationTestModelLoaded<PackageNo>
{
protected:
  using ValidationTestSingleSession::_session;

  void SetUp() override
  {
    ValidationTestModelLoaded<PackageNo>::SetUp();
    nnfw_prepare(_session);
  }

  void TearDown() override { ValidationTestModelLoaded<PackageNo>::TearDown(); }

  void SetInOutBuffers()
  {
    nnfw_tensorinfo ti_input;
    ASSERT_EQ(nnfw_input_tensorinfo(_session, 0, &ti_input), NNFW_STATUS_NO_ERROR);
    uint64_t input_elements = num_elems(&ti_input);
    EXPECT_EQ(input_elements, 1);
    _input.resize(input_elements);
    ASSERT_EQ(
      nnfw_set_input(_session, 0, ti_input.dtype, _input.data(), sizeof(float) * input_elements),
      NNFW_STATUS_NO_ERROR);

    nnfw_tensorinfo ti_output;
    ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &ti_output), NNFW_STATUS_NO_ERROR);
    uint64_t output_elements = num_elems(&ti_output);
    EXPECT_EQ(output_elements, 1);
    _output.resize(output_elements);
    ASSERT_EQ(nnfw_set_output(_session, 0, ti_output.dtype, _output.data(),
                              sizeof(float) * output_elements),
              NNFW_STATUS_NO_ERROR);
  }

  void SetInOutBuffersDynamic(const nnfw_tensorinfo *ti_input)
  {
    NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(_session, 0, ti_input));
    uint64_t input_elements = num_elems(ti_input);
    _input.resize(input_elements);
    ASSERT_EQ(
      nnfw_set_input(_session, 0, ti_input->dtype, _input.data(), sizeof(float) * input_elements),
      NNFW_STATUS_NO_ERROR);

    _output.resize(40000); // Give sufficient size for the output
    ASSERT_EQ(
      nnfw_set_output(_session, 0, ti_input->dtype, _output.data(), sizeof(float) * _output.size()),
      NNFW_STATUS_NO_ERROR);
  }

protected:
  std::vector<float> _input;
  std::vector<float> _output;
};

template <int PackageNo> class ValidationTestFourModelsSetInput : public ValidationTest
{
protected:
  static const uint32_t NUM_SESSIONS = 4;

  void SetUp() override
  {
    ValidationTest::SetUp();

    for (auto &obj : _objects)
    {
      ASSERT_EQ(nnfw_create_session(&obj.session), NNFW_STATUS_NO_ERROR);

      auto cbuf = genAddModel();
      NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(obj.session, cbuf.buffer(), cbuf.size()));
      ASSERT_EQ(nnfw_prepare(obj.session), NNFW_STATUS_NO_ERROR);
      _cbufs.push_back(std::move(cbuf)); // Keep the buffer so it can outlive the session

      uint32_t num_inputs;
      ASSERT_EQ(nnfw_input_size(obj.session, &num_inputs), NNFW_STATUS_NO_ERROR);
      obj.inputs.resize(num_inputs);
      for (uint32_t ind = 0; ind < obj.inputs.size(); ind++)
      {
        nnfw_tensorinfo ti;
        ASSERT_EQ(nnfw_input_tensorinfo(obj.session, ind, &ti), NNFW_STATUS_NO_ERROR);
        uint64_t input_elements = num_elems(&ti);
        obj.inputs[ind].resize(input_elements);
        ASSERT_EQ(nnfw_set_input(obj.session, ind, ti.dtype, obj.inputs[ind].data(),
                                 sizeof(float) * input_elements),
                  NNFW_STATUS_NO_ERROR);
      }

      uint32_t num_outputs;
      ASSERT_EQ(nnfw_output_size(obj.session, &num_outputs), NNFW_STATUS_NO_ERROR);
      obj.outputs.resize(num_outputs);
      for (uint32_t ind = 0; ind < obj.outputs.size(); ind++)
      {
        nnfw_tensorinfo ti;
        ASSERT_EQ(nnfw_output_tensorinfo(obj.session, ind, &ti), NNFW_STATUS_NO_ERROR);
        uint64_t output_elements = num_elems(&ti);
        obj.outputs[ind].resize(output_elements);
        ASSERT_EQ(nnfw_set_output(obj.session, ind, ti.dtype, obj.outputs[ind].data(),
                                  sizeof(float) * output_elements),
                  NNFW_STATUS_NO_ERROR);
      }
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
  std::vector<CircleBuffer> _cbufs;
};

class ValidationTestTwoSessions : public ValidationTest
{
protected:
  nnfw_session *_session1 = nullptr;
  nnfw_session *_session2 = nullptr;
};

class ValidationTestTwoSessionsCreated : public ValidationTestTwoSessions
{
protected:
  void SetUp() override
  {
    ValidationTestTwoSessions::SetUp();
    ASSERT_EQ(nnfw_create_session(&_session1), NNFW_STATUS_NO_ERROR);
    ASSERT_EQ(nnfw_create_session(&_session2), NNFW_STATUS_NO_ERROR);
    ASSERT_NE(_session1, nullptr);
    ASSERT_NE(_session2, nullptr);
  }

  void TearDown() override
  {
    ASSERT_EQ(nnfw_close_session(_session1), NNFW_STATUS_NO_ERROR);
    ASSERT_EQ(nnfw_close_session(_session2), NNFW_STATUS_NO_ERROR);
    ValidationTestTwoSessions::TearDown();
  }
};

class ValidationTestPipelineSession : public ValidationTest
{
protected:
  nnfw_session *_session = nullptr;
};

#endif // __NNFW_API_TEST_FIXTURES_H__
