/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExecTime.h"

#include "backend/IConfig.h"
#include "backend/Backend.h"

#include <gtest/gtest.h>

#include <string>

namespace
{
using namespace onert;
using namespace exec;
using namespace backend;

struct MockConfig : public IConfig
{
  std::string id() override { return "b1"; }
  bool initialize() override { return true; };
  bool supportDynamicTensor() override { return false; }
  bool supportFP16() override { return false; }
};

struct MockBackend : public ::onert::backend::Backend
{
  std::shared_ptr<onert::backend::IConfig> config() const override
  {
    return std::make_shared<MockConfig>();
  }
  std::unique_ptr<onert::backend::BackendContext> newContext(ContextData &&) const override
  {
    return nullptr;
  }
};

TEST(ExecTime, roundtrip_ok)
{
  const auto *b = new MockBackend();
  std::vector<const Backend *> bs = {b};
  {
    ExecTime et(bs);
    et.updateOperationExecTime(b, "op1", true, 100, 100);
    et.updateOperationExecTime(b, "op1", true, 200, 200);
    et.updateOperationExecTime(b, "op1", false, 100, 888);
    et.storeOperationsExecTime();
  }
  {
    ExecTime et(bs);
    auto time = et.getOperationExecTime(b, "op1", true, 100);
    ASSERT_EQ(time, 100);
    // Check interpolation
    time = et.getOperationExecTime(b, "op1", true, 150);
    ASSERT_EQ(time, 150);
    time = et.getOperationExecTime(b, "op1", false, 100);
    ASSERT_EQ(time, 888);
    et.storeOperationsExecTime();
  }
  // clean up
  EXPECT_EQ(remove("exec_time.json"), 0);
}

TEST(ExecTime, structure)
{

  const auto *b = new MockBackend();
  std::vector<const Backend *> bs = {b};
  {
    ExecTime et(bs);
    et.updateOperationExecTime(b, "op1", true, 100, 100);
    et.updateOperationExecTime(b, "op1", true, 200, 200);
    et.storeOperationsExecTime();
  }
  {
    ExecTime et(bs);
    auto time = et.getOperationExecTime(b, "op1", true, 100);
    ASSERT_EQ(time, 100);
    // Check interpolation
    time = et.getOperationExecTime(b, "op1", true, 200);
    ASSERT_EQ(time, 200);
    et.storeOperationsExecTime();
  }
  // clean up
  EXPECT_EQ(remove("exec_time.json"), 0);
}
} // unnamed namespace
