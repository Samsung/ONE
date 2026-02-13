/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "../BulkPipelineManager.h"
#include <gtest/gtest.h>

#include "mock_syscalls.h"

using namespace onert::backend::trix::ops;
using namespace onert::backend::trix::ops::test;

class BulkPipelineManagerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    BulkPipelineManager::PipelineConfig config;
    config.device_id = 0;
    config.model_paths.push_back("model_path");
    manager = std::make_unique<BulkPipelineManager>(config);

    // Reset all mock syscalls before each test
    MockSyscallsManager::getInstance().resetAll();

    MockSyscallsManager::getInstance().setFreadHook(
      [](void *ptr, size_t size, size_t, FILE *) -> int {
        if (size == NPUBIN_META_SIZE)
        {
          auto meta = reinterpret_cast<npubin_meta *>(ptr);
          meta->program_size = 1024;
          meta->weight_size = 1024;
          meta->size = 4096;
        }
        return 1;
      });

    MockSyscallsManager::getInstance().setIoctlHook(
      [](int, unsigned long request, void *arg) -> int {
        // Get Version
        if (request == _IOR(0x88, 1, unsigned int))
        {
          // Return version 3.2.X.X for trix backend sanity checking
          *static_cast<int *>(arg) = 0x3020000;
        }
        return 0;
      });
  }
  void TearDown() override {}

  std::unique_ptr<BulkPipelineManager> manager;
  const int nr_models = 1;
};

TEST_F(BulkPipelineManagerTest, test_initilize)
{
  EXPECT_TRUE(manager->initialize());
  EXPECT_TRUE(manager->isInitialized());
}

TEST_F(BulkPipelineManagerTest, test_shutdown)
{
  int nr_fclose_calls = 0;
  EXPECT_TRUE(manager->initialize());
  // This hook will checking the number of fclose() calls
  MockSyscallsManager::getInstance().setFcloseHook([&nr_fclose_calls](FILE *) -> int {
    nr_fclose_calls++;
    return 0;
  });
  manager->shutdown();
  EXPECT_FALSE(manager->isInitialized());
  // fclose() should be called as the same number of models
  EXPECT_EQ(nr_fclose_calls, nr_models);
}

TEST_F(BulkPipelineManagerTest, test_execute)
{
  EXPECT_TRUE(manager->initialize());
  const std::vector<const onert::backend::IPortableTensor *> inputs;
  std::vector<onert::backend::IPortableTensor *> outputs;
  EXPECT_NO_THROW(manager->execute(inputs, outputs));
}
