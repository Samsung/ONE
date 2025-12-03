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

#include "../BulkPipelineBuffer.h"
#include <gtest/gtest.h>

#include "mock_syscalls.h"

using namespace onert::backend::trix::ops;

class BulkPipelineBufferTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Create a standard buffer for testing
    buffer =
      std::make_unique<BulkPipelineBuffer>(BulkPipelineBuffer::BufferType::DMABUF_CONT, 1024, 0);
  }

  void TearDown() override
  {
    // Ensure buffer is properly deallocated
    if (buffer && buffer->isReady())
    {
      buffer->deallocate();
    }
  }

  std::unique_ptr<BulkPipelineBuffer> buffer;
};

TEST_F(BulkPipelineBufferTest, test_allocate)
{
  EXPECT_NO_THROW(buffer->allocate());
  EXPECT_TRUE(buffer->isReady());
  EXPECT_EQ(buffer->size(), 1024);
}

TEST_F(BulkPipelineBufferTest, test_deallocate)
{
  buffer->allocate();
  buffer->deallocate();
  EXPECT_FALSE(buffer->isReady());
  EXPECT_EQ(buffer->size(), 0);
}

TEST_F(BulkPipelineBufferTest, test_fillFromFile)
{
  auto dummy_fp = fopen("/dev/null", "r");
  ASSERT_NE(dummy_fp, nullptr) << "Failed to open /dev/null for testing";

  EXPECT_ANY_THROW(buffer->fillFromFile(nullptr, 0));

  buffer->allocate();
  EXPECT_NO_THROW(buffer->fillFromFile(dummy_fp, 0));
  buffer->deallocate();

  fclose(dummy_fp);
}
