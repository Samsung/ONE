/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "args.h"

#include <iostream>

namespace TFLiteRun
{

Args::Args(const int argc, char **argv) noexcept
  : _arser("Load tflite model by onert and TFLite, and compare their output")
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  // positional argument
  _arser.add_argument("tflite")
    .type(arser::DataType::STR)
    .help("Input tflite model file for serialization");

  // optional argument
  _arser.add_argument("--data", "-d")
    .type(arser::DataType::STR)
    .accumulated()
    .help("Input data file for model");
}

void Args::print(char **) { std::cout << _arser; }

void Args::Parse(const int argc, char **argv)
{
  try
  {
    _arser.parse(argc, argv);

    if (_arser["tflite"])
    {
      _tflite_filename = _arser.get<std::string>("tflite");
    }

    if (_arser["--data"])
    {
      _data_filenames = _arser.get<std::vector<std::string>>("--data");
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    print(argv);
    exit(1);
  }
}

} // end of namespace TFLiteRun
