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

#include "hermes/core/Severity.h"

#include <gtest/gtest.h>

TEST(SeverityTest, fatal)
{
  auto severity = hermes::fatal();

  ASSERT_EQ(severity.category(), hermes::FATAL);
  ASSERT_EQ(severity.level(), 0);
}

TEST(SeverityTest, error)
{
  auto severity = hermes::error();

  ASSERT_EQ(severity.category(), hermes::ERROR);
  ASSERT_EQ(severity.level(), 0);
}

TEST(SeverityTest, warn)
{
  auto severity = hermes::warn();

  ASSERT_EQ(severity.category(), hermes::WARN);
  ASSERT_EQ(severity.level(), 0);
}

TEST(SeverityTest, info)
{
  auto severity = hermes::info();

  ASSERT_EQ(severity.category(), hermes::INFO);
  ASSERT_EQ(severity.level(), 0);
}

TEST(SeverityTest, verbose)
{
  auto severity = hermes::verbose(100);

  ASSERT_EQ(severity.category(), hermes::VERBOSE);
  ASSERT_EQ(severity.level(), 100);
}
