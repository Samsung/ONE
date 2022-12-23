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
  iface->handle_context_create = &on_handle_context_create;
  iface->handle_context_destroy = &on_handle_context_destroy;
  iface->handle_buffers_create = &on_handle_buffers_create;
  iface->handle_buffers_destroy = &on_handle_buffers_destroy;
  iface->handle_network_create = &on_handle_network_create;
  iface->handle_network_destroy = &on_handle_network_destroy;
  iface->handle_request_create = &on_handle_request_create;
  iface->handle_request_destroy = &on_handle_request_destroy;
  iface->handle_request_set_data = &on_handle_request_set_data;
  iface->handle_execute_run = &on_handle_execute_run;

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
  std::vector<std::string> list;
  int error = Server::instance().core().getAvailableDeviceList(list);
  // TODO Implement variant outputs
  npud_core_complete_device_get_available_list(object, invocation, error);
  return TRUE;
}

gboolean DBus::on_handle_context_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                        gint arg_device_id, gint arg_priority)
{
  VERBOSE(DBus) << "on_handle_context_create with " << arg_device_id << ", " << arg_priority
                << std::endl;

  guint64 out_ctx = 0;
  int ret = Server::instance().core().createContext(arg_device_id, arg_priority, &out_ctx);
  npud_core_complete_context_create(object, invocation, out_ctx, ret);
  return TRUE;
}

gboolean DBus::on_handle_context_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                         guint64 arg_ctx)
{
  VERBOSE(DBus) << "on_handle_context_destroy with " << arg_ctx << std::endl;
  int ret = Server::instance().core().destroyContext(arg_ctx);
  npud_core_complete_context_destroy(object, invocation, ret);
  return TRUE;
}

gboolean DBus::on_handle_buffers_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                        guint64 arg_ctx, GVariant *arg_buffers)
{
  VERBOSE(DBus) << "on_handle_buffers_create with " << arg_ctx << std::endl;
  GenericBuffers bufs;
  GVariantIter *iter = NULL;
  gint32 type;
  guint64 addr;
  guint32 size;
  int index = 0;
  g_variant_get(arg_buffers, "a(itu)", &iter);
  while (iter != NULL && g_variant_iter_loop(iter, "(itu)", &type, &addr, &size))
  {
    VERBOSE(DBus) << "in [" << index << "] Type: " << type << ", Addr: " << addr
                  << ", Size: " << size << std::endl;
    bufs.buffers[index].type = static_cast<BufferTypes>(type);
    bufs.buffers[index].addr = reinterpret_cast<void *>(addr);
    bufs.buffers[index].size = size;
    index++;
  }
  bufs.numBuffers = index;
  g_variant_iter_free(iter);

  // TODO Invoke Core function.
  int ret = -1;

  GVariantBuilder *builder = g_variant_builder_new(G_VARIANT_TYPE("a(itu)"));

  // TODO Enable below code when we can update ret value by core function
  // if (ret == 0)
  // {
  //   for (auto i = 0; i < bufs.numBuffers; ++i)
  //   {
  //     VERBOSE(DBus) << "out [" << index << "] Type: " << bufs.buffers[i].type
  //                   << ", Addr: " << bufs.buffers[i].addr << ", Size: " << bufs.buffers[i].size
  //                   << std::endl;
  //     g_variant_builder_add(builder, "(itu)", bufs.buffers[i].type, bufs.buffers[i].addr,
  //                           bufs.buffers[i].size);
  //   }
  // }
  npud_core_complete_buffers_create(object, invocation, g_variant_builder_end(builder), ret);
  return TRUE;
}

gboolean DBus::on_handle_buffers_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                         guint64 arg_ctx, GVariant *arg_buffers)
{
  VERBOSE(DBus) << "on_handle_buffers_destroy with " << arg_ctx << std::endl;
  GenericBuffers bufs;
  GVariantIter *iter = NULL;
  gint32 type;
  guint64 addr;
  guint32 size;
  int index = 0;
  g_variant_get(arg_buffers, "a(itu)", &iter);
  while (iter != NULL && g_variant_iter_loop(iter, "(itu)", &type, &addr, &size))
  {
    VERBOSE(DBus) << "[" << index << "] Type: " << type << ", Addr: " << (void *)addr
                  << ", Size: " << size << std::endl;
    bufs.buffers[index].type = static_cast<BufferTypes>(type);
    bufs.buffers[index].addr = reinterpret_cast<void *>(addr);
    bufs.buffers[index].size = size;
    index++;
  }
  bufs.numBuffers = index;
  g_variant_iter_free(iter);
  // TODO Invoke Core function.
  int ret = -1;
  npud_core_complete_buffers_destroy(object, invocation, ret);
  return TRUE;
}

gboolean DBus::on_handle_network_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                        guint64 arg_ctx, const gchar *arg_model_path)
{
  VERBOSE(DBus) << "on_handle_network_create with " << arg_ctx << ", " << arg_model_path
                << std::endl;
  ModelID modelId = 0;
  int ret = Server::instance().core().createNetwork(arg_ctx, arg_model_path, &modelId);
  npud_core_complete_network_create(object, invocation, guint(modelId), ret);
  return TRUE;
}

gboolean DBus::on_handle_network_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                         guint64 arg_ctx, guint arg_nw_handle)
{
  VERBOSE(DBus) << "on_handle_network_destroy with " << arg_ctx << ", " << arg_nw_handle
                << std::endl;
  int ret = Server::instance().core().destroyNetwork(arg_ctx, arg_nw_handle);
  npud_core_complete_network_destroy(object, invocation, ret);
  return TRUE;
}

gboolean DBus::on_handle_request_create(NpudCore *object, GDBusMethodInvocation *invocation,
                                        guint64 arg_ctx, guint arg_nw_handle)
{
  VERBOSE(DBus) << "on_handle_request_create with " << arg_ctx << ", " << arg_nw_handle
                << std::endl;
  RequestID requestID = 0;
  int ret = Server::instance().core().createRequest(arg_ctx, arg_nw_handle, &requestID);
  npud_core_complete_request_create(object, invocation, guint(requestID), ret);
  return TRUE;
}

gboolean DBus::on_handle_request_destroy(NpudCore *object, GDBusMethodInvocation *invocation,
                                         guint64 arg_ctx, guint arg_rq_handle)
{
  VERBOSE(DBus) << "on_handle_request_destroy with " << arg_ctx << ", " << arg_rq_handle
                << std::endl;
  int ret = Server::instance().core().destroyRequest(arg_ctx, arg_rq_handle);
  npud_core_complete_request_destroy(object, invocation, ret);
  return TRUE;
}

gboolean DBus::on_handle_request_set_data(NpudCore *object, GDBusMethodInvocation *invocation,
                                          guint64 arg_ctx, guint arg_rq_handle,
                                          GVariant *arg_input_buffers, GVariant *arg_output_buffers)
{
  VERBOSE(DBus) << "on_handle_request_set_data with " << arg_ctx << ", " << arg_rq_handle
                << std::endl;
  GVariantIter *iter = NULL;
  InputBuffers inBufs;
  OutputBuffers outBufs;
  gint32 type;
  guint64 addr;
  guint32 size;
  int index = 0;

  // inBufs
  g_variant_get(arg_input_buffers, "a(itu)", &iter);
  index = 0;
  while (iter != NULL && g_variant_iter_loop(iter, "(itu)", &type, &addr, &size))
  {
    VERBOSE(DBus) << "in [" << index << "] Type: " << type << ", Addr: " << (void *)addr
                  << ", Size: " << size << std::endl;
    if (type == 0) // NPU_BUFFER_MAPPED
    {
      inBufs.buffers[index].addr = reinterpret_cast<void *>(addr);
    }
    else if (type == 1) // NPU_BUFFER_DMABUF
    {
      // TODO Support dma buffer
      VERBOSE(DBus) << "[NYI] NPU_BUFFER_DMABUF" << std::endl;
      continue;
    }
    else
    {
      VERBOSE(DBus) << "Wrong buffer type. Ignored." << std::endl;
      continue;
    }
    inBufs.buffers[index].size = size;
    inBufs.buffers[index].type = static_cast<BufferTypes>(type);
    index++;
  }
  inBufs.numBuffers = index;
  g_variant_iter_free(iter);

  // outBufs
  g_variant_get(arg_output_buffers, "a(itu)", &iter);
  index = 0;
  while (iter != NULL && g_variant_iter_loop(iter, "(itu)", &type, &addr, &size))
  {
    VERBOSE(DBus) << "out [" << index << "] Type: " << type << ", Addr: " << (void *)addr
                  << ", Size: " << size << std::endl;
    if (type == 0) // NPU_BUFFER_MAPPED
    {
      outBufs.buffers[index].addr = reinterpret_cast<void *>(addr);
    }
    else if (type == 1) // NPU_BUFFER_DMABUF
    {
      // TODO Support dma buffer
      VERBOSE(DBus) << "[NYI] NPU_BUFFER_DMABUF" << std::endl;
      continue;
    }
    else
    {
      VERBOSE(DBus) << "Wrong buffer type. Ignored." << std::endl;
      continue;
    }
    outBufs.buffers[index].size = size;
    outBufs.buffers[index].type = static_cast<BufferTypes>(type);
    index++;
  }
  outBufs.numBuffers = index;
  g_variant_iter_free(iter);

  // TODO Invoke Core function.
  int ret = -1;
  npud_core_complete_request_set_data(object, invocation, ret);
  return TRUE;
}

gboolean DBus::on_handle_execute_run(NpudCore *object, GDBusMethodInvocation *invocation,
                                     guint64 arg_ctx, guint arg_rq_handle)
{
  VERBOSE(DBus) << "on_handle_execute_run with " << arg_ctx << ", " << arg_rq_handle << std::endl;
  // TODO Invoke Core function.
  int ret = -1;
  npud_core_complete_execute_run(object, invocation, ret);
  return TRUE;
}

} // namespace core
} // namespace npud
