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

#include "../BulkPipelineModel.h"
#include <gtest/gtest.h>

#include "mock_syscalls.h"

using namespace onert::backend::trix::ops;
using namespace onert::backend::trix::ops::test;

class BulkPipelineModelTest : public ::testing::Test
{

protected:
  void SetUp() override
  {
    model = std::make_unique<BulkPipelineModel>("model_path", 0);

    // Reset all mock syscalls before each test
    MockSyscallsManager::getInstance().resetAll();

    // Add a hook for fread()
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

  void TearDown() override
  {
    // Clear all mock syscalls after each test
    MockSyscallsManager::getInstance().resetAll();
  }

  std::unique_ptr<BulkPipelineModel> model;
};

// Mock trix-engine api
int registerNPUmodel_ext(npudev_h, generic_buffer *, generic_buffer *, generic_buffer *,
                         uint32_t *model_id)
{
  *model_id = 1;
  return 0;
}

int runNPU_model(npudev_h, uint32_t, npu_infer_mode, const input_buffers *, output_buffers *,
                 npuOutputNotify, void *)
{
  return 0;
}

int unregisterNPUmodel(npudev_h, uint32_t) { return 0; }

TEST_F(BulkPipelineModelTest, test_model_creation)
{
  EXPECT_TRUE(model->initialize());
  EXPECT_TRUE(model->prepare());

  EXPECT_NE(model->metadata(), nullptr);
  EXPECT_EQ(model->programSize(), 1024);
  EXPECT_EQ(model->weightSize(), 1024);
  EXPECT_NE(model->device(), nullptr);
  EXPECT_NE(model->modelId(), 0);
  EXPECT_EQ(model->modelPath(), "model_path");
}

TEST_F(BulkPipelineModelTest, test_model_run)
{
  EXPECT_TRUE(model->initialize());
  EXPECT_TRUE(model->prepare());
  const std::vector<const onert::backend::IPortableTensor *> inputs;
  std::vector<onert::backend::IPortableTensor *> outputs;
  EXPECT_NO_THROW(model->run(inputs, outputs));
}

TEST_F(BulkPipelineModelTest, test_model_release)
{
  EXPECT_TRUE(model->initialize());
  EXPECT_TRUE(model->prepare());
  model->release();
  EXPECT_EQ(model->device(), nullptr);
  EXPECT_EQ(model->modelId(), 0);
  EXPECT_EQ(model->metadata(), nullptr);
}

TEST_F(BulkPipelineModelTest, test_async_fill)
{
  model->initialize();
  model->prepare();

  std::shared_ptr<BulkPipelineModel> next_model;
  next_model = std::make_shared<BulkPipelineModel>("next_model_path", 0);
  next_model->initialize();
  next_model->prepare();
  next_model->setBufferOwnership(BulkPipelineModel::BufferOwnership::SHARED);

  model->setNextModel(next_model);

  EXPECT_NO_THROW(next_model->startAsyncBufferFill());
  EXPECT_NO_THROW(next_model->waitForBufferReady());

  // Release model for negative test
  next_model->release();
  // Exception will be thrown to different thread
  next_model->startAsyncBufferFill();
  EXPECT_ANY_THROW(next_model->waitForBufferReady());
}
