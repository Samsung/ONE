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

#include <gtest/gtest.h>
#include <iostream>

namespace
{
using namespace npud;
using namespace core;

// //
// // Mock backends classes
// //

// struct MockServer : public Server
// {
// public:
//   MOCK_METHOD(void, run, (), (override));
//   MOCK_METHOD(void, stop, (), (override));
//   MOCK_METHOD(std::atomic_bool&, isRunning, (), (override));
// };

//
// Tests setup/teardown
//

// SetUp/TearDown methods runs before/after each test and performs actions common for each test
class ServerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::cout << "SetUp" << std::endl;
    childPid = fork();
    if (childPid == -1) {
      std::cout << "Fork error" << std::endl;
    } else if (childPid == 0) {
      // child
      std::cout << "Child: " << "I'm child" << std::endl;
      auto &server = Server::instance();
      server.run();
    } else {
      // parent
      std::cout << "Check child" << std::endl;
      auto &server = Server::instance();
      while(true) {
        std::cout << "Server status: " << server.isRunning() << std::endl;
        sleep(1);
        if (server.isRunning() == true) {
          break;
        }
      }
      std::cout << "Child is running" << std::endl;
    }
  }

  void TearDown() override
  {
    std::cout << "TearDown" << std::endl;
    auto &server = Server::instance();
    server.stop();

    int status;
    pid_t w = waitpid(childPid, &status, 0);
    if (w == -1) {
      std::cout << "Error on waitpid" << std::endl;
    }

    if (WIFEXITED(status)) {
      std::cout << "Child is exited" << std::endl;
    } else {
      std::cout << "Child status: " << status << std::endl;
    }
  }

  pid_t childPid;
};

//
// Server tests
//

// Test scheduler behavior for straight graph with known execution time of all nodes and permutes.
TEST_F(ServerTest, run)
{
  auto &server = Server::instance();
  ASSERT_EQ(server.isRunning(), true);
}

// TODO: Add tests with unknown execution and permutation time

} // unnamed namespace
