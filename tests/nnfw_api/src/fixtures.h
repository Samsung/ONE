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

#include <gtest/gtest.h>
#include <nnfw.h>

#include "model_path.h"

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

#endif // __NNFW_API_TEST_FIXTURES_H__
