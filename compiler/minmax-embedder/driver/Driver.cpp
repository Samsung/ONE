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

#include "minmax-embedder/Embedder.h"

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <stdlib.h>

using namespace minmax_embedder;

void print_version(void)
{
  std::cout << "minmax-embedder version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(const int argc, char **argv)
{
  arser::Arser arser("minmax-embedder embeds given minmax into circle");
  arser::Helper::add_version(arser, print_version);
  // named args
  arser.add_argument("--min_percentile")
    .type(arser::DataType::FLOAT)
    .default_value(1.f)
    .help("Set min percentile (default: 1)");
  arser.add_argument("--max_percentile")
    .type(arser::DataType::FLOAT)
    .default_value(99.f)
    .help("Set max percentile (default: 99)");
  arser.add_argument("-o").default_value("out.circle").help("Path to output circle model");
  // positional args: minmax(h5), input(circle)
  arser.add_argument("circle").help("Path to input circle model");
  arser.add_argument("minmax").help("Path to minmax data in hdf5");
  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  std::string minmax_path = arser.get<std::string>("minmax");
  std::string circle_path = arser.get<std::string>("circle");
  std::string output_path = arser.get<std::string>("-o");
  float min_percentile = arser.get<float>("--min_percentile");
  float max_percentile = arser.get<float>("--max_percentile");

  EmbedderOptions opt{min_percentile, max_percentile};
  try
  {
    Embedder().embed(output_path, circle_path, minmax_path, opt);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
