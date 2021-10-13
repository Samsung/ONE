/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <arser/arser.h>

#include "CircleModel.h"
#include "TFLModel.h"

#include <vconone/vconone.h>

void print_version(void)
{
  std::cout << "tflite2circle version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  arser::Arser arser{"tflite2circle is a Tensorflow lite to circle model converter"};

  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  arser.add_argument("-V", "--verbose")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("output additional information to stdout or stderr");

  arser.add_argument("tflite")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Source tflite file path to convert");
  arser.add_argument("circle").nargs(1).type(arser::DataType::STR).help("Target circle file path");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  std::string tfl_path = arser.get<std::string>("tflite");
  std::string circle_path = arser.get<std::string>("circle");
  // read tflite file
  tflite2circle::TFLModel tfl_model(tfl_path);

  // create flatbuffer builder
  auto flatbuffer_builder = std::make_unique<flatbuffers::FlatBufferBuilder>(1024);

  // convert tflite to circle
  tflite2circle::CircleModel circle_model{flatbuffer_builder, tfl_model.get_model()};

  std::ofstream outfile{circle_path, std::ios::binary};

  outfile.write(circle_model.base(), circle_model.size());
  outfile.close();
  // TODO find a better way of error handling
  if (outfile.fail())
  {
    std::cerr << "ERROR: Failed to write circle '" << circle_path << "'" << std::endl;
    return 255;
  }

  return 0;
}
