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
