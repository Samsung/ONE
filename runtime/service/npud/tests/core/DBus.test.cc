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

#include <core/Server.h>
#include <gtest/gtest.h>
#include <thread>
#include <gio/gio.h>
#include <dbus-core.h>
#include <iostream>

namespace
{
using namespace npud;
using namespace core;

//
// DBusTest setup/teardown
//
class DBusTest : public ::testing::Test
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
    while (server.isServiceReady() != true)
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

  NpudCore *getProxy()
  {
    GError *error = nullptr;
    NpudCore *proxy = nullptr;
    proxy = npud_core_proxy_new_for_bus_sync(G_BUS_TYPE_SYSTEM, G_DBUS_PROXY_FLAGS_NONE,
                                             "org.tizen.npud", "/org/tizen/npud", NULL, &error);
    if (error)
    {
      g_error_free(error);
    }
    return proxy;
  }
};

//
// DBusTest
//
TEST_F(DBusTest, get_proxy)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);
}

TEST_F(DBusTest, device_get_available_list)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  npud_core_call_device_get_available_list_sync(proxy, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

} // unnamed namespace
