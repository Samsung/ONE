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

#include "Server.h"

#include <gio/gunixfdlist.h>
#include <thread>
#include <util/Logging.h>

namespace npud
{
namespace core
{

std::atomic_bool Server::_isRunning(false);

Server::Server() noexcept
  : _mainloop(g_main_loop_new(NULL, FALSE), g_main_loop_unref), _signal(std::make_unique<Signal>())
{
}

void Server::run(void)
{
  VERBOSE(Server) << "Starting Server\n";

  if (_isRunning.exchange(true))
  {
    throw std::runtime_error("Mainloop is already running.");
  }

  _gdbus_id = g_bus_own_name(G_BUS_TYPE_SYSTEM,
                          "org.tizen.npud", G_BUS_NAME_OWNER_FLAGS_NONE,
                          (GBusAcquiredCallback)on_bus_acquired,
                          (GBusNameAcquiredCallback)on_name_acquired,
                          (GBusNameLostCallback)on_name_lost,
                          NULL, NULL);
  g_main_loop_run(_mainloop.get());
}

void Server::stop(void)
{
  VERBOSE(Server) << "Stop Server\n";

  if (!_isRunning.load())
  {
    throw std::runtime_error("Mainloop is not running");
  }

  while (!g_main_loop_is_running(_mainloop.get()))
  {
    std::this_thread::yield();
  }

  g_bus_unown_name(_gdbus_id);
  g_main_loop_quit(_mainloop.get());
  _isRunning = false;
}

gboolean Server::on_handle_device_get_available_list(NpudCore *core,
GDBusMethodInvocation *invocation,
guint seconds,
gpointer user_data)
{
  VERBOSE(Server) << "on_handle_device_get_available_list" << std::endl;
  VERBOSE(Server) << core << std::endl;
}

gboolean Server::emit_alarm_cb(gpointer core)
{
  npud_core_emit_beep(NPUD_CORE(core));
  npud_core_set_activated(NPUD_CORE(core), FALSE);
  return FALSE;
}

gboolean Server::on_handle_configure(NpudCore *core,
GDBusMethodInvocation *invocation,
guint seconds,
gpointer user_data)
{
  VERBOSE(Server) << "on_handle_configure" << std::endl;
  
  if (npud_core_get_activated(core)) {
    g_dbus_method_invocation_return_error_literal(invocation, G_IO_ERROR, G_IO_ERROR_EXISTS, "Exists");
    return false;
  }

  npud_core_set_activated(core, TRUE);
  g_timeout_add_seconds(seconds, emit_alarm_cb, core);
  npud_core_complete_configure(core, invocation);
}

gboolean Server::on_handle_context_create(NpudCore *object,
    GDBusMethodInvocation *invocation,
    gint arg_device_id,
    gint arg_priority)
{
  VERBOSE(Server) << "on_handle_context_create" << std::endl;
}

void Server::on_bus_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
    VERBOSE(Server) << "on bus acquired" << std::endl;

  GError *error = NULL;
  NpudCore *core = npud_core_skeleton_new();

  g_signal_connect(core, "handle-context-create", G_CALLBACK(on_handle_context_create), NULL);
  g_signal_connect(core, "handle-configure", G_CALLBACK(on_handle_configure), NULL);
  g_signal_connect(core, "handle-device-get-available-list", G_CALLBACK(on_handle_device_get_available_list), user_data);

  if (!g_dbus_interface_skeleton_export(G_DBUS_INTERFACE_SKELETON(core), conn, "/org/tizen/npud", &error)) {
    VERBOSE(Server) << "[ERROR] export skeleton" << std::endl;
  }
}

void Server::on_name_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(Server) << "on name acquired" << std::endl;
}

void Server::on_name_lost(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(Server) << "on name lost" << std::endl;
}

} // namespace core
} // namespace npud
