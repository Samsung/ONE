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

#include "args.h"

#include <iostream>

namespace TFLiteRun220
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
    // General options
    po::options_description general("General options");

    // clang-format off
  general.add_options()
    ("help,h", "Display available options")
    ("input,i", po::value<std::string>()->default_value(""), "Input filename")
    ("dump,d", po::value<std::string>()->default_value(""), "Output filename")
    ("ishapes", po::value<std::vector<int>>()->multitoken(), "Input shapes")
    ("compare,c", po::value<std::string>()->default_value(""), "filename to be compared with")
    ("tflite", po::value<std::string>()->required())
    ("num_runs,r", po::value<int>()->default_value(1), "The number of runs")
    ("warmup_runs,w", po::value<int>()->default_value(0), "The number of warmup runs")
    ("run_delay,t", po::value<int>()->default_value(-1), "Delay time(ms) between runs (as default no delay")
    ("gpumem_poll,g", po::value<bool>()->default_value(false), "Check gpu memory polling separately")
    ("mem_poll,m", po::value<bool>()->default_value(false), "Check memory polling")
    ("write_report,p", po::value<bool>()->default_value(false), "Write report")
    ("validate", po::value<bool>()->default_value(true), "Validate tflite model")
    ("verbose_level,v", po::value<int>()->default_value(0), "Verbose level\n"
         "0: prints the only result. Messages btw run don't print\n"
         "1: prints result and message btw run\n"
         "2: prints all of messages to print\n")
    ;
    // clang-format on

    _options.add(general);
    _positional.add("tflite", 1);
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "error by bad cast during initialization of boost::program_options" << e.what()
              << '\n';
    exit(1);
  }
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);

  {
    auto conflicting_options = [&](const std::string &o1, const std::string &o2) {
      if ((vm.count(o1) && !vm[o1].defaulted()) && (vm.count(o2) && !vm[o2].defaulted()))
      {
        throw boost::program_options::error(std::string("Two options '") + o1 + "' and '" + o2 +
                                            "' cannot be given at once.");
      }
    };

    conflicting_options("input", "compare");
  }

  if (vm.count("help"))
  {
    std::cout << "tflite_run\n\n";
    std::cout << "Usage: " << argv[0] << " <.tflite> [<options>]\n\n";
    std::cout << _options;
    std::cout << "\n";

    exit(0);
  }

  po::notify(vm);

  if (vm.count("dump"))
  {
    _dump_filename = vm["dump"].as<std::string>();
  }

  if (vm.count("compare"))
  {
    _compare_filename = vm["compare"].as<std::string>();
  }

  if (vm.count("input"))
  {
    _input_filename = vm["input"].as<std::string>();

    if (!_input_filename.empty())
    {
      if (access(_input_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "input image file not found: " << _input_filename << "\n";
      }
    }
  }

  if (vm.count("ishapes"))
  {
    _input_shapes.resize(vm["ishapes"].as<std::vector<int>>().size());
    for (auto i = 0; i < _input_shapes.size(); i++)
    {
      _input_shapes[i] = vm["ishapes"].as<std::vector<int>>()[i];
    }
  }

  if (vm.count("tflite"))
  {
    _tflite_filename = vm["tflite"].as<std::string>();

    if (_tflite_filename.empty())
    {
      // TODO Print usage instead of the below message
      std::cerr << "Please specify tflite file. Run with `--help` for usage."
                << "\n";

      exit(1);
    }
    else
    {
      if (access(_tflite_filename.c_str(), F_OK) == -1)
      {
        std::cerr << "tflite file not found: " << _tflite_filename << "\n";
        exit(1);
      }
    }
  }

  if (vm.count("num_runs"))
  {
    _num_runs = vm["num_runs"].as<int>();
  }

  if (vm.count("warmup_runs"))
  {
    _warmup_runs = vm["warmup_runs"].as<int>();
  }

  if (vm.count("run_delay"))
  {
    _run_delay = vm["run_delay"].as<int>();
  }

  if (vm.count("gpumem_poll"))
  {
    _gpumem_poll = vm["gpumem_poll"].as<bool>();
  }

  if (vm.count("mem_poll"))
  {
    _mem_poll = vm["mem_poll"].as<bool>();
    // Instead of EXECUTE to avoid overhead, memory polling runs on WARMUP
    if (_mem_poll && _warmup_runs == 0)
    {
      _warmup_runs = 1;
    }
  }

  if (vm.count("write_report"))
  {
    _write_report = vm["write_report"].as<bool>();
  }

  if (vm.count("validate"))
  {
    _tflite_validate = vm["validate"].as<bool>();
  }

  if (vm.count("verbose_level"))
  {
    _verbsoe_level = vm["verbose_level"].as<int>();
  }
}

} // end of namespace TFLiteRun220
