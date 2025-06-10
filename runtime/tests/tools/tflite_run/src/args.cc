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

#include "args.h"

#include <unistd.h>

namespace TFLiteRun
{

Args::Args(const int argc, char **argv) noexcept
{
  try
  {
    Initialize();
    Parse(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << "error during paring args" << e.what() << '\n';
    exit(1);
  }
}

void Args::Initialize(void)
{
  try
  {
    _arser.add_argument("--input", "-i")
      .type(arser::DataType::STR)
      .default_value("")
      .help("Input filename");
    _arser.add_argument("--dump", "-d")
      .type(arser::DataType::STR)
      .default_value("")
      .help("Output filename");
    _arser.add_argument("--ishapes").type(arser::DataType::INT32_VEC).help("Input shapes");
    _arser.add_argument("--compare", "-c")
      .type(arser::DataType::STR)
      .default_value("")
      .help("Filename to be compared with");
    _arser.add_argument("--num_runs", "-r")
      .type(arser::DataType::INT32)
      .default_value(1)
      .help("The number of runs");
    _arser.add_argument("--warmup_runs", "-w")
      .type(arser::DataType::INT32)
      .default_value(0)
      .help("The number of warmup runs");
    _arser.add_argument("--run_delay", "-t")
      .type(arser::DataType::INT32)
      .default_value(-1)
      .help("Delay time(ms) between runs (as default no delay)");
    _arser.add_argument("--gpumem_poll", "-g")
      .nargs(0)
      .default_value(false)
      .help("Check gpu memory polling separately");
    _arser.add_argument("--mem_poll", "-m")
      .nargs(0)
      .default_value(false)
      .help("Check memory polling");
    _arser.add_argument("--write_report", "-p").nargs(0).default_value(false).help("Write report");
    _arser.add_argument("--verbose_level", "-v")
      .type(arser::DataType::INT32)
      .default_value(0)
      .help("Verbose level\n"
            "0: prints the only result. Messages btw run don't print\n"
            "1: prints result and message btw run\n"
            "2: prints all of messages to print\n");
    _arser.add_argument("tflite").type(arser::DataType::STR);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << _arser;
    exit(1);
  }
}

void Args::Parse(const int argc, char **argv)
{

  try
  {
    _arser.parse(argc, argv);

    // Get parsed options
    _input_filename = _arser.get<std::string>("--input");
    _dump_filename = _arser.get<std::string>("--dump");
    _input_shapes = _arser.get<std::vector<int>>("--ishapes");
    _compare_filename = _arser.get<std::string>("--compare");
    _num_runs = _arser.get<int>("--num_runs");
    _warmup_runs = _arser.get<int>("--warmup_runs");
    _run_delay = _arser.get<int>("--run_delay");
    _gpumem_poll = _arser.get<bool>("--gpumem_poll");
    _mem_poll = _arser.get<bool>("--mem_poll");
    _write_report = _arser.get<bool>("--write_report");
    _verbose_level = _arser.get<int>("--verbose_level");
    _tflite_filename = _arser.get<std::string>("tflite");

    // Validation check
    if (!_input_filename.empty())
    {
      if (access(_input_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "Input image file not found: " << _input_filename << std::endl;
        exit(1);
      }
    }

    if (!_tflite_filename.empty())
    {
      if (access(_tflite_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "TFLite file not found: " << _tflite_filename << std::endl;
        exit(1);
      }
    }

    // Check conflict option
    if (!_input_filename.empty() && !_compare_filename.empty())
    {
      std::cerr << "Two options '--input' and '--compare' cannot be given at once" << std::endl;
      exit(1);
    }

    // Instead of EXECUTE to avoid overhead, memory polling runs on WARMUP
    if (_mem_poll && _warmup_runs == 0)
    {
      _warmup_runs = 1;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    exit(1);
  }
}

} // end of namespace TFLiteRun
