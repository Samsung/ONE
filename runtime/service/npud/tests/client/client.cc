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
#include <thread>
#include <csignal>

#include "Model.h"
#include "Request.h"

// GMainLoop *gmain;

// void handleSignal(int signum)
// {
//   std::cout << "Signal received: " << signum << std::endl;
//   while (!g_main_loop_is_running(gmain))
//   {
//     std::this_thread::yield();
//   }

//   g_main_loop_quit(gmain);
// }

int main(int argc, char *argv[])
{
  using namespace npud::tests::client;

  // std::signal(SIGTERM, handleSignal);

  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " [model path]" << std::endl;
    return 1;
  }

  std::string model_path = argv[1];
  if (access(model_path.c_str(), F_OK) != 0)
  {
    std::cout << "[ERROR] Invalid model path: " << model_path << std::endl;
    return 1;
  }

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

  if (request.execute_run(ctx, rq_handle) != 0)
  {
    std::cout << "Failed to execute run" << std::endl;
  }

  // gmain = g_main_loop_new(NULL, FALSE);
  // g_main_loop_run(gmain);

  // std::cout << "Bye~~" << std::endl;
  request.buffers_destroy(ctx, inbufs);
  request.buffers_destroy(ctx, outbufs);
  request.request_destroy(ctx, rq_handle);
  request.network_destroy(ctx, nw_handle);
  request.context_destroy(ctx);
  return 0;
}
