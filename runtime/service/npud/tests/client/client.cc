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

#include <string>

#include "Model.h"
#include "Request.h"

int main(int argc, char *argv[])
{
  using namespace npud::tests::client;

  std::string model_path = "./test-models/model.tvn";
  Model model(model_path);
  Request request;

  gint device_id = 0;
  gint priority = 0;
  guint64 ctx = 0;
  if (request.context_create(device_id, priority, &ctx) != 0)
  {
    return 1;
  }

  guint nw_handle = 0;
  if (request.network_create(ctx, model.get_name(), &nw_handle) != 0)
  {
    request.context_destroy(ctx);
    return 1;
  }

  input_buffers *inbufs = model.get_inputs();
  output_buffers *outbufs = model.get_outputs();
  if (request.buffers_create(ctx, inbufs) != 0 || request.buffers_create(ctx, outbufs) != 0)
  {
    request.context_destroy(ctx);
    return 1;
  }

  guint rq_handle = 0;
  if (request.request_create(ctx, nw_handle, &rq_handle) != 0)
  {
    request.network_destroy(ctx, nw_handle);
    request.context_destroy(ctx);
    return 1;
  }

  if (request.request_set_data(ctx, rq_handle, inbufs, outbufs) != 0)
  {
    request.request_destroy(ctx, rq_handle);
    request.network_destroy(ctx, nw_handle);
    request.context_destroy(ctx);
    return 1;
  }

  return 0;
}
