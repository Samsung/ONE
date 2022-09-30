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

#include "DBus.h"

#include <atomic>
#include <util/Logging.h>

namespace npud
{
namespace core
{

std::atomic_bool DBus::_isReady(false);

DBus::DBus() noexcept
{
  VERBOSE(DBus) << "Starting dbus service" << std::endl;

  _gdbus_id = g_bus_own_name(G_BUS_TYPE_SYSTEM, "org.tizen.npud", G_BUS_NAME_OWNER_FLAGS_NONE,
                             (GBusAcquiredCallback)on_bus_acquired,
                             (GBusNameAcquiredCallback)on_name_acquired,
                             (GBusNameLostCallback)on_name_lost, NULL, NULL);
}

DBus::~DBus() noexcept
{
  VERBOSE(DBus) << "Stop dbus service" << std::endl;

  g_bus_unown_name(_gdbus_id);
}

void DBus::on_bus_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(DBus) << "on bus acquired" << std::endl;

  GError *error = NULL;
  NpudCore *core = npud_core_skeleton_new();

  g_signal_connect(core, "handle-device-get-available-list",
                   G_CALLBACK(on_handle_device_get_available_list), user_data);
  g_signal_connect(core, "handle-context-create", G_CALLBACK(on_handle_context_create), NULL);

  if (!g_dbus_interface_skeleton_export(G_DBUS_INTERFACE_SKELETON(core), conn, "/org/tizen/npud",
                                        &error))
  {
    VERBOSE(DBus) << "[ERROR] export skeleton" << std::endl;
  }

  _isReady.exchange(true);
}

void DBus::on_name_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(DBus) << "on name acquired" << std::endl;
}

void DBus::on_name_lost(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(DBus) << "on name lost" << std::endl;
}

gboolean DBus::on_handle_device_get_available_list(NpudCore *object,
                                                   GDBusMethodInvocation *invocation,
                                                   GUnixFDList *fd_list)
{
  VERBOSE(DBus) << "on_handle_device_get_available_list" << std::endl;

  GUnixFDList *out_fd_list = NULL;
  // TODO Implement details
  npud_core_complete_device_get_available_list(object, invocation, out_fd_list, 0);
  return TRUE;
}

gboolean DBus::on_handle_context_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                        gint arg_device_id, gint arg_priority)
{
  VERBOSE(DBus) << "on_handle_context_create with " << arg_device_id << ", " << arg_priority
                << std::endl;
  // TODO Implement details
  guint64 out_ctx = 100;
  npud_core_complete_context_create(object, invocation, out_ctx);
  return TRUE;
}

} // namespace core
} // namespace npud
