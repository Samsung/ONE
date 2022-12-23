/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONE_SERVICE_NPUD_CORE_DBUS_H__
#define __ONE_SERVICE_NPUD_CORE_DBUS_H__

#include <dbus-core.h>
#include <gio/gio.h>
#include <memory>
#include <atomic>

namespace npud
{
namespace core
{

class DBus
{
public:
  DBus() noexcept;
  ~DBus() noexcept;

  DBus(const DBus &) = delete;
  DBus &operator=(const DBus &) = delete;

  bool isReady() { return _isReady.load(); }

  static void on_bus_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data);
  static void on_name_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data);
  static void on_name_lost(GDBusConnection *conn, const gchar *name, gpointer user_data);

  static gboolean on_handle_device_get_available_list(NpudCore *object,
                                                      GDBusMethodInvocation *invocation);
  static gboolean on_handle_context_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                           gint arg_device_id, gint arg_priority);
  static gboolean on_handle_context_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                            guint64 arg_ctx);
  static gboolean on_handle_buffers_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                           guint64 arg_ctx, GVariant *arg_buffers);
  static gboolean on_handle_buffers_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                            guint64 arg_ctx, GVariant *arg_buffers);
  static gboolean on_handle_network_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                           guint64 arg_ctx, const gchar *arg_model_path);
  static gboolean on_handle_network_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                            guint64 arg_ctx, guint arg_nw_handle);
  static gboolean on_handle_request_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                           guint64 arg_ctx, guint arg_nw_handle);
  static gboolean on_handle_request_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                            guint64 arg_ctx, guint arg_rq_handle);
  static gboolean on_handle_request_set_data(NpudCore *object, GDBusMethodInvocation *invocation,
                                             guint64 arg_ctx, guint arg_rq_handle,
                                             GVariant *arg_input_buffers,
                                             GVariant *arg_output_buffers);
  static gboolean on_handle_execute_run(NpudCore *object, GDBusMethodInvocation *invocation,
                                        guint64 arg_ctx, guint arg_rq_handle);

private:
  guint _gdbus_id;
  static std::atomic_bool _isReady;
};

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_DBUS_H__
