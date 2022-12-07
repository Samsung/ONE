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

#ifndef __ONE_SERVICE_NPUD_TEST_CLIENT_REQUEST_H__
#define __ONE_SERVICE_NPUD_TEST_CLIENT_REQUEST_H__

#include <iostream>
#include <dbus-core.h>

namespace npud
{
namespace tests
{
namespace client
{

class Request
{
public:
  Request()
  {
    GError *error = nullptr;
    auto proxy =
      npud_core_proxy_new_for_bus_sync(G_BUS_TYPE_SYSTEM, G_DBUS_PROXY_FLAGS_NONE, "org.tizen.npud",
                                       "/org/tizen/npud", NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      throw std::runtime_error("failed to get proxy");
    }

    _proxy = proxy;
  }

  ~Request() {}

  int context_create(gint device_id, gint priority, guint64 *ctx)
  {
    GError *error = nullptr;
    gint out_error = -1;
    guint64 out_ctx = 0;
    npud_core_call_context_create_sync(_proxy, device_id, priority, &out_ctx, &out_error, NULL,
                                       &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }

    *ctx = out_ctx;
    return 0;
  }

  int context_destroy(guint64 ctx)
  {
    GError *error = nullptr;
    gint out_error = -1;
    npud_core_call_context_destroy_sync(_proxy, ctx, &out_error, NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }
    return 0;
  }

  int buffers_create(guint64 ctx, generic_buffers *bufs)
  {
    GError *error = nullptr;
    gint out_error = -1;
    GVariantBuilder *builder = g_variant_builder_new(G_VARIANT_TYPE("a(itu)"));
    GVariant *out_bufs;
    for (int i = 0; i < bufs->num_buffers; ++i)
    {
      std::cout << "in [" << i << "] Type: " << bufs->bufs[i].type
                << ", Addr: " << (void *)bufs->bufs[i].addr << ", Size: " << bufs->bufs[i].size
                << std::endl;
      g_variant_builder_add(builder, "(itu)", bufs->bufs[i].type, bufs->bufs[i].addr,
                            bufs->bufs[i].size);
    }
    npud_core_call_buffers_create_sync(_proxy, ctx, g_variant_builder_end(builder), &out_bufs,
                                       &out_error, NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }

    GVariantIter *iter = NULL;
    g_variant_get(out_bufs, "a(itu)", &iter);
    gint type;
    guint64 addr;
    guint32 size;
    int index = 0;
    while (iter != NULL && g_variant_iter_loop(iter, "(itu)", &type, &addr, &size))
    {
      std::cout << "out [" << index << "] Type: " << type << ", Addr: " << (void *)addr
                << ", Size: " << size << std::endl;
      bufs->bufs[index].type = static_cast<buffer_types>(type);
      bufs->bufs[index].addr = reinterpret_cast<void *>(addr);
      bufs->bufs[index].size = size;
      index++;
    }
    bufs->num_buffers = index;
    g_variant_iter_free(iter);
    return 0;
  }

  int buffers_destroy(guint64 ctx, generic_buffers *bufs)
  {
    GError *error = nullptr;
    gint out_error = -1;
    GVariantBuilder *builder = g_variant_builder_new(G_VARIANT_TYPE("a(itu)"));
    for (int i = 0; i < bufs->num_buffers; ++i)
    {
      g_variant_builder_add(builder, "(itu)", bufs->bufs[i].type, bufs->bufs[i].addr,
                            bufs->bufs[i].size);
    }
    npud_core_call_buffers_destroy_sync(_proxy, ctx, g_variant_builder_end(builder), &out_error,
                                        NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }
    memset(bufs, '/x00', sizeof(generic_buffers));
    return 0;
  }

  int network_create(guint64 ctx, std::string name, guint *nw_handle)
  {
    GError *error = nullptr;
    gint out_error = -1;
    guint out_nw_handle = 0;
    npud_core_call_network_create_sync(_proxy, ctx, name.c_str(), &out_nw_handle, &out_error, NULL,
                                       &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }

    *nw_handle = out_nw_handle;
    return 0;
  }

  int network_destroy(guint64 ctx, guint nw_handle)
  {
    GError *error = nullptr;
    gint out_error = -1;
    npud_core_call_network_destroy_sync(_proxy, ctx, nw_handle, &out_error, NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }
    return 0;
  }

  int request_create(guint64 ctx, guint nw_handle, guint *rq_handle)
  {
    GError *error = nullptr;
    gint out_error = -1;
    guint out_rq_handle = 0;
    npud_core_call_request_create_sync(_proxy, ctx, nw_handle, &out_rq_handle, &out_error, NULL,
                                       &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }

    *rq_handle = out_rq_handle;
    return 0;
  }

  int request_destroy(guint64 ctx, guint rq_handle)
  {
    GError *error = nullptr;
    gint out_error = -1;
    npud_core_call_request_destroy_sync(_proxy, ctx, rq_handle, &out_error, NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }
    return 0;
  }

  int request_set_data(guint64 ctx, guint rq_handle, input_buffers *inbufs, output_buffers *outbufs)
  {
    GError *error = nullptr;
    gint out_error = -1;
    if (inbufs == nullptr || outbufs == nullptr)
    {
      std::cout << "Invalid buffers" << std::endl;
      return 1;
    }

    GVariantBuilder *input_builder = g_variant_builder_new(G_VARIANT_TYPE("a(itu)"));
    for (int i = 0; i < inbufs->num_buffers; ++i)
    {
      g_variant_builder_add(input_builder, "(itu)", inbufs->bufs[i].type, inbufs->bufs[i].addr,
                            inbufs->bufs[i].size);
    }
    GVariantBuilder *output_builder = g_variant_builder_new(G_VARIANT_TYPE("a(itu)"));
    for (int i = 0; i < outbufs->num_buffers; ++i)
    {
      g_variant_builder_add(output_builder, "(itu)", outbufs->bufs[i].type, outbufs->bufs[i].addr,
                            outbufs->bufs[i].size);
    }
    npud_core_call_request_set_data_sync(
      _proxy, ctx, rq_handle, g_variant_builder_end(input_builder),
      g_variant_builder_end(output_builder), &out_error, NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }
    return 0;
  }

  int execute_run(guint64 ctx, guint rq_handle)
  {
    GError *error = nullptr;
    gint out_error = -1;
    npud_core_call_execute_run_sync(_proxy, ctx, rq_handle, &out_error, NULL, &error);
    if (error)
    {
      std::cout << error->message << std::endl;
      g_error_free(error);
      return 1;
    }
    return 0;
  }

private:
  NpudCore *_proxy;
};

} // namespace client
} // namespace tests
} // namespace npud

#endif // __ONE_SERVICE_NPUD_TEST_CLIENT_REQUEST_H__
