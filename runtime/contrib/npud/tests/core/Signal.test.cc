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

#include "core/Server.h"
#include "core/Signal.h"

#include <gtest/gtest.h>
#include <thread>
#include <csignal>

namespace
{
using namespace npud;
using namespace core;

//
// SignalTest setup/teardown
//
class SignalTest : public ::testing::Test
{
protected:
  static void runTask()
  {
    auto &server = Server::instance();
    server.run();
  }

  void SetUp() override
  {
    std::thread child = std::thread(runTask);
    child.detach();
    auto &server = Server::instance();
    while (server.isRunning() != true)
    {
    }
  }

  void TearDown() override
  {
    auto &server = Server::instance();
    if (server.isRunning())
    {
      server.stop();
    }
  }
};

//
// SignalTest
//
TEST_F(SignalTest, raise_SIGTERM)
{
  auto &server = Server::instance();
  ASSERT_EQ(server.isRunning(), true);
  std::raise(SIGTERM);
  ASSERT_EQ(server.isRunning(), false);
}

TEST_F(SignalTest, raise_SIGINT)
{
  auto &server = Server::instance();
  ASSERT_EQ(server.isRunning(), true);
  std::raise(SIGINT);
  ASSERT_EQ(server.isRunning(), false);
}

TEST_F(SignalTest, raise_SIGILL)
{
  auto &server = Server::instance();
  ASSERT_EQ(server.isRunning(), true);
  std::raise(SIGILL);
  ASSERT_EQ(server.isRunning(), false);
}

TEST_F(SignalTest, raise_SIGABRT)
{
  auto &server = Server::instance();
  ASSERT_EQ(server.isRunning(), true);
  std::raise(SIGABRT);
  ASSERT_EQ(server.isRunning(), false);
}

TEST_F(SignalTest, raise_SIGFPE)
{
  auto &server = Server::instance();
  ASSERT_EQ(server.isRunning(), true);
  std::raise(SIGFPE);
  ASSERT_EQ(server.isRunning(), false);
}

} // unnamed namespace
