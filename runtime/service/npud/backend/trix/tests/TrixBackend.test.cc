/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "../TrixBackend.h"

#include <gtest/gtest.h>

namespace
{
using namespace npud;
using namespace backend;
using namespace trix;

//
// TrixBackendTest setup/teardown
//
class TrixBackendTest : public ::testing::Test
{
protected:
  void SetUp() override { _trix.createContext(0, 0, &_ctx); }

  void TearDown() override { _trix.destroyContext(_ctx); }

  TrixBackend _trix;
  NpuContext *_ctx;
};

//
// TrixBackendTest
//
TEST_F(TrixBackendTest, createContext)
{
  TrixBackend trixbackend;
  NpuContext *ctx;
  auto ret = trixbackend.createContext(0, 0, &ctx);
  ASSERT_EQ(ret, NPU_STATUS_SUCCESS);
}

TEST_F(TrixBackendTest, createContext_invalid_id_NEG)
{
  TrixBackend trixbackend;
  NpuContext *ctx;
  auto ret = trixbackend.createContext(-1, 0, &ctx);
  ASSERT_EQ(ret, NPU_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F(TrixBackendTest, createContext_invalid_priority_NEG)
{
  // TODO Need to enable.
  GTEST_SKIP();
  TrixBackend trixbackend;
  NpuContext *ctx;
  auto ret = trixbackend.createContext(0, -1, &ctx);
  ASSERT_EQ(ret, NPU_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F(TrixBackendTest, destroyContext)
{
  TrixBackend trixbackend;
  NpuContext *ctx;
  auto ret = trixbackend.createContext(0, 0, &ctx);
  ASSERT_EQ(ret, NPU_STATUS_SUCCESS);

  ret = trixbackend.destroyContext(ctx);
  ASSERT_EQ(ret, NPU_STATUS_SUCCESS);
}

TEST_F(TrixBackendTest, destroyContext_invalid_ctx_NEG)
{
  TrixBackend trixbackend;
  NpuContext *ctx = nullptr;

  auto ret = trixbackend.destroyContext(ctx);
  ASSERT_EQ(ret, NPU_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F(TrixBackendTest, registerModel)
{
  // TODO Use valid model.
  const std::string modelPath = "abc.model";
  ModelID id;
  auto ret = _trix.registerModel(_ctx, modelPath, &id);
  ASSERT_EQ(ret, NPU_STATUS_SUCCESS);
}

TEST_F(TrixBackendTest, registerModel_wrong_model_NEG)
{
  const std::string modelPath = "wrong.model";
  ModelID id;
  auto ret = _trix.registerModel(_ctx, modelPath, &id);
  ASSERT_EQ(ret, NPU_STATUS_ERROR_OPERATION_FAILED);
}

TEST_F(TrixBackendTest, unregisterModel)
{
  // TODO Use valid model.
  const std::string modelPath = "abc.model";
  ModelID id;
  auto ret = _trix.registerModel(_ctx, modelPath, &id);
  ASSERT_EQ(ret, NPU_STATUS_SUCCESS);

  ret = _trix.unregisterModel(_ctx, id);
  ASSERT_EQ(ret, NPU_STATUS_SUCCESS);
}

TEST_F(TrixBackendTest, unregisterModel_invalid_id_NEG)
{
  ModelID id = 0;
  auto ret = _trix.unregisterModel(_ctx, id);
  ASSERT_EQ(ret, NPU_STATUS_ERROR_INVALID_MODEL);
}

} // unnamed namespace
