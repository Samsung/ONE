/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "H5Writer.h"

#include <arser/arser.h>

using namespace minmax_embedder_test;

int entry(const int argc, char **argv)
{
  arser::Arser arser("Generate min/max data to test minmax-embedder");
  arser.add_argument("--num_inputs")
    .type(arser::DataType::INT32)
    .default_value(1)
    .help("number of input layers (default:1)");
  arser.add_argument("--num_ops")
    .type(arser::DataType::INT32)
    .default_value(1)
    .help("number of operators (default:1)");
  arser.add_argument("minmax").help("path to generated minmax data");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  auto num_inputs = arser.get<int>("--num_inputs");
  auto num_ops = arser.get<int>("--num_ops");
  auto data_output_path = arser.get<std::string>("minmax");

  ModelSpec mspec;
  {
    if (num_inputs <= 0 || num_ops <= 0)
    {
      std::cout << "num_inputs and num_ops must be positive integers." << std::endl;
      return 255;
    }
    mspec.n_inputs = num_inputs;
    mspec.n_ops = num_ops;
  }
  H5Writer writer(mspec, data_output_path);
  writer.dump();

  return EXIT_SUCCESS;
}
