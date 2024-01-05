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

#include <CLI11.hpp>
#include <iostream>

namespace TFLiteRun
{

Args::Args(const int argc, char **argv) noexcept
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  // General options
  _app =
    std::make_shared<CLI::App>("Load TFLite model by onert and TFLite, and compare their output");

  _app->add_option("--tflite", _tflite_filename, "Input tflite model file for serialization")
    ->type_name("PATH")
    ->check(CLI::ExistingFile);
  _app->add_option("--data,-d", _data_filenames, "Input data file for model")
    ->type_name("PATH")
    ->check(CLI::ExistingFile);

  // Positional (higher priority)
  _app->add_option("modelfile", _tflite_filename, "Input tflite model file for serialization")
    ->type_name("PATH")
    ->check(CLI::ExistingFile);
}

void Args::print() { _app->exit(CLI::CallForHelp()); }

void Args::Parse(const int argc, char **argv)
{
  try
  {
    _app->parse(argc, argv);
  }
  catch (const CLI::ParseError &e)
  {
    exit(_app->exit(e));
  }
}

} // end of namespace TFLiteRun
