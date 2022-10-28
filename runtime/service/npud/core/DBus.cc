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
#include "Server.h"

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
  VERBOSE(DBus) << name << " on bus acquired" << std::endl;

  GError *error = NULL;
  NpudCore *core = npud_core_skeleton_new();
  NpudCoreIface *iface = NPUD_CORE_GET_IFACE(core);

  iface->handle_device_get_available_list = &on_handle_device_get_available_list;

  if (!g_dbus_interface_skeleton_export(G_DBUS_INTERFACE_SKELETON(core), conn, "/org/tizen/npud",
                                        &error))
  {
    VERBOSE(DBus) << "Failed to export skeleton, Server will stop." << std::endl;
    Server::instance().stop();
  }

  _isReady.exchange(true);
}

void DBus::on_name_acquired(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(DBus) << name << " on name acquired" << std::endl;
}

void DBus::on_name_lost(GDBusConnection *conn, const gchar *name, gpointer user_data)
{
  VERBOSE(DBus) << name << " on name lost, Server will stop." << std::endl;
  Server::instance().stop();
}

gboolean DBus::on_handle_device_get_available_list(NpudCore *object,
                                                   GDBusMethodInvocation *invocation)
{
  VERBOSE(DBus) << __FUNCTION__ << std::endl;
  // TODO Implement details
  int error = 0;
  npud_core_complete_device_get_available_list(object, invocation, error);
  return TRUE;
}

} // namespace core
} // namespace npud
