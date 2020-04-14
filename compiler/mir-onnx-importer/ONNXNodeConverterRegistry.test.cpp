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

#include "ONNXNodeConverterRegistry.h"
#include "ONNXHelpers.h"

#include "gtest/gtest.h"

using namespace mir_onnx;

void converterV1(const onnx::NodeProto &node, ConverterContext *ctx) {}
void converterV3(const onnx::NodeProto &node, ConverterContext *ctx) {}
void converterV7(const onnx::NodeProto &node, ConverterContext *ctx) {}

class NodeConverterRegsitryTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    registry.registerConverter("dummy", 1, converterV1);
    registry.registerConverter("dummy", 3, converterV3);
    registry.registerConverter("dummy", 7, converterV7);
    registry.registerConverter("dummy", firstUnknownOpset, nullptr);
  }

  NodeConverterRegistry registry;
};

TEST_F(NodeConverterRegsitryTest, existing_lookup_works)
{
  auto res = registry.lookup("dummy", 1);
  ASSERT_EQ(res, &converterV1);
}

TEST_F(NodeConverterRegsitryTest, skipped_lookup_works)
{
  auto res = registry.lookup("dummy", 2);
  ASSERT_EQ(res, &converterV1);
}

TEST_F(NodeConverterRegsitryTest, first_unknown_version_works)
{
  auto res = registry.lookup("dummy", 14);
  ASSERT_EQ(res, nullptr);
}

TEST_F(NodeConverterRegsitryTest, lower_than_first_version)
{
  auto res = registry.lookup("dummy", 0);
  ASSERT_EQ(res, nullptr);
}
