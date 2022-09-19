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

#ifndef __ONE_SERVICE_NPUD_CORE_SERVER_H__
#define __ONE_SERVICE_NPUD_CORE_SERVER_H__

#include "Signal.h"
#include "dbus-core.h"

#include <glib.h>
#include <gio/gio.h>
#include <memory>
#include <atomic>

namespace npud
{
namespace core
{

class Server
{
public:
  void run(void);
  void stop(void);

  bool isRunning() { return _isRunning.load(); }

  static Server &instance(void)
  {
    static Server server;
    return server;
  }

  static void on_bus_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data);
  static void on_name_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data);
  static void on_name_lost(GDBusConnection *conn, const gchar *name, gpointer user_data);

  static gboolean on_handle_device_get_available_list(NpudCore *core,
    GDBusMethodInvocation *invocation,
    guint seconds,
    gpointer user_data);
  static gboolean emit_alarm_cb(gpointer core);
  static gboolean on_handle_configure(NpudCore *core,
    GDBusMethodInvocation *invocation,
    guint seconds,
    gpointer user_data);
  static gboolean on_handle_context_create(NpudCore *object,
    GDBusMethodInvocation *invocation,
    gint arg_device_id,
    gint arg_priority);

private:
  Server() noexcept;

  static std::atomic_bool _isRunning;

  std::unique_ptr<GMainLoop, void (*)(GMainLoop *)> _mainloop;
  std::unique_ptr<Signal> _signal;
  guint _gdbus_id;
};

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_SERVER_H__
