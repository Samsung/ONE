/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <tensorflow/core/public/session.h>

#include <iostream>
#include <stdexcept>

#include <cassert>
#include <cstring>

#include <benchmark/Accumulator.h>

#define CHECK_TF(e)                                \
  {                                                \
    if (!(e).ok())                                 \
    {                                              \
      throw std::runtime_error{"'" #e "' FAILED"}; \
    }                                              \
  }

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "USAGE: " << argv[0] << " [T/F model path] [output 0] [output 1] ..." << std::endl;
    return 255;
  }

  std::vector<std::string> output_nodes;

  for (int argn = 2; argn < argc; ++argn)
  {
    output_nodes.emplace_back(argv[argn]);
  }

  tensorflow::Session *sess;

  CHECK_TF(tensorflow::NewSession(tensorflow::SessionOptions(), &sess));

  tensorflow::GraphDef graph_def;

  CHECK_TF(ReadBinaryProto(tensorflow::Env::Default(), argv[1], &graph_def));
  CHECK_TF(sess->Create(graph_def));

  tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 320, 320, 3}));
  std::vector<tensorflow::Tensor> outputs;

  for (uint32_t n = 0; n < 5; ++n)
  {
    std::chrono::milliseconds elapsed(0);

    benchmark::measure(elapsed) << [&](void) {
      CHECK_TF(sess->Run({{"input_node", input}}, output_nodes, {}, &outputs));
    };

    std::cout << "Takes " << elapsed.count() << "ms" << std::endl;
  }

  return 0;
}
