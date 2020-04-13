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

#include "hermes/core/Message.h"

#include <gtest/gtest.h>

TEST(MessageTextTest, multiline)
{
  std::stringstream ss;

  ss << "Hello, World" << std::endl;
  ss << "Nice to meet you" << std::endl;

  hermes::MessageText text{ss};

  ASSERT_EQ(text.lines(), 2);
  ASSERT_EQ(text.line(0), "Hello, World");
  ASSERT_EQ(text.line(1), "Nice to meet you");
}

TEST(MessageTest, ctor)
{
  hermes::Message msg;

  // Text is empty at the beginning
  ASSERT_EQ(msg.text(), nullptr);
}
