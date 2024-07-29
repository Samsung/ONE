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

  const std::string &getModel()
  {
    if (model.empty())
    {
      auto model_path = std::getenv("GTEST_MODEL_PATH");
      model = model_path + std::string("/mv1.q8/mv1.q8.tvn");
    }
    if (access(model.c_str(), F_OK) != 0)
    {
      model.clear();
    }
    return model;
  }

private:
  std::string model;
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

TEST_F(DBusTest, context_create)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

TEST_F(DBusTest, context_destroy)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);

  npud_core_call_context_destroy_sync(proxy, out_ctx, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

TEST_F(DBusTest, neg_context_destroy_invalid_ctx)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  guint64 out_ctx = 0;
  npud_core_call_context_destroy_sync(proxy, out_ctx, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, network_create)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

TEST_F(DBusTest, neg_network_create_invalid_ctx)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  guint64 out_ctx = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, neg_network_create_invalid_model)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  // Invalid model
  const gchar *model_path = "invalid.tvn";
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, network_destroy)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  npud_core_call_network_destroy_sync(proxy, out_ctx, out_nw_handle, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

TEST_F(DBusTest, neg_network_destroy_invalid_ctx)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  // Invalid ctx
  out_ctx = -1;
  npud_core_call_network_destroy_sync(proxy, out_ctx, out_nw_handle, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, neg_network_destroy_invalid_nw_handle)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  guint out_nw_handle = -1;
  npud_core_call_network_destroy_sync(proxy, out_ctx, out_nw_handle, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, request_create)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  guint out_rq_handle = 0;
  npud_core_call_request_create_sync(proxy, out_ctx, out_nw_handle, &out_rq_handle, &out_error,
                                     NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

TEST_F(DBusTest, neg_request_create_invalid_ctx)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  guint out_rq_handle = 0;
  npud_core_call_request_create_sync(proxy, 0, out_nw_handle, &out_rq_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, neg_request_create_invalid_nw)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  guint out_rq_handle = 0;
  npud_core_call_request_create_sync(proxy, out_ctx, 0, &out_rq_handle, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, request_destroy)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  guint out_rq_handle = 0;
  npud_core_call_request_create_sync(proxy, out_ctx, out_nw_handle, &out_rq_handle, &out_error,
                                     NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  npud_core_call_request_destroy_sync(proxy, out_ctx, out_rq_handle, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);
}

TEST_F(DBusTest, neg_request_destroy_invalid_ctx)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  const gchar *model_path = this->getModel().c_str();
  guint out_nw_handle = 0;
  npud_core_call_network_create_sync(proxy, out_ctx, model_path, &out_nw_handle, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  guint out_rq_handle = 0;
  npud_core_call_request_create_sync(proxy, out_ctx, out_nw_handle, &out_rq_handle, &out_error,
                                     NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  npud_core_call_request_destroy_sync(proxy, 0, out_rq_handle, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

TEST_F(DBusTest, neg_request_destroy_invalid_rq)
{
  NpudCore *proxy = this->getProxy();
  ASSERT_NE(proxy, nullptr);

  GError *error = NULL;
  gint out_error = -1;
  gint arg_device_id = 0;
  gint arg_priority = 0;
  guint64 out_ctx = 0;
  npud_core_call_context_create_sync(proxy, arg_device_id, arg_priority, &out_ctx, &out_error, NULL,
                                     &error);
  if (error)
  {
    g_error_free(error);
    error = NULL;
  }
  ASSERT_EQ(out_error, 0);

  out_error = -1;
  npud_core_call_request_destroy_sync(proxy, out_ctx, 0, &out_error, NULL, &error);
  if (error)
  {
    g_error_free(error);
  }
  ASSERT_NE(out_error, 0);
}

} // unnamed namespace
