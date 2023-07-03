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

#include "args.h"

#include <functional>
#include <iostream>
#include <sys/stat.h>
#include <json/json.h>

namespace
{

// This function parses a json object and returns as a vector of integers
// For example,
// [0, [1, 2, 3, 4], 3, 40, 4, []] in JSON
// is converted to:
// {
//  0 -> [1, 2, 3, 4]
//  3 -> 40
//  4 -> []
// } in std::unordered_map. Note that the value type is still Json::Value.
std::unordered_map<uint32_t, Json::Value> argArrayToMap(const Json::Value &jsonval)
{
  if (!jsonval.isArray() || (jsonval.size() % 2 != 0))
  {
    std::cerr << "JSON argument must be an even-sized array in JSON\n";
    exit(1);
  }

  std::unordered_map<uint32_t, Json::Value> ret;
  for (uint32_t i = 0; i < jsonval.size(); i += 2)
  {
    if (!jsonval[i].isUInt())
    {
      std::cerr << "Key values(values in even indices) must be unsigned integers\n";
      exit(1);
    }
    uint32_t key = jsonval[i].asUInt();
    Json::Value val = jsonval[i + 1];
    ret[key] = jsonval[i + 1];
  }
  return ret;
}

void checkModelfile(const std::string &model_filename)
{
  if (model_filename.empty())
  {
    // TODO Print usage instead of the below message
    std::cerr << "Please specify model file. Run with `--help` for usage."
              << "\n";

    exit(1);
  }
  else
  {
    if (access(model_filename.c_str(), F_OK) == -1)
    {
      std::cerr << "Model file not found: " << model_filename << "\n";
      exit(1);
    }
  }
}

void checkPackage(const std::string &package_filename)
{
  if (package_filename.empty())
  {
    // TODO Print usage instead of the below message
    std::cerr << "Please specify nnpackage file. Run with `--help` for usage."
              << "\n";

    exit(1);
  }
  else
  {
    if (access(package_filename.c_str(), F_OK) == -1)
    {
      std::cerr << "nnpackage not found: " << package_filename << "\n";
      exit(1);
    }
  }
}

} // namespace

namespace onert_train
{

Args::Args(const int argc, char **argv)
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  auto process_nnpackage = [&](const std::string &package_filename) {
    _package_filename = package_filename;

    std::cerr << "Package Filename " << _package_filename << std::endl;
    checkPackage(package_filename);
  };

  auto process_modelfile = [&](const std::string &model_filename) {
    _model_filename = model_filename;

    std::cerr << "Model Filename " << _model_filename << std::endl;
    checkModelfile(model_filename);

    _use_single_model = true;
  };

  auto process_inputdata = [&](const std::string &input_filename) {
    _input_filename = input_filename;

    std::cerr << "Model Input Filename " << _input_filename << std::endl;
    checkModelfile(_input_filename);
  };

  auto process_expecteddata = [&](const std::string &expected_filename) {
    _expected_filename = expected_filename;

    std::cerr << "Model Expected Filename " << _expected_filename << std::endl;
    checkModelfile(_expected_filename);
  };

  auto process_path = [&](const std::string &path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0)
    {
      if (sb.st_mode & S_IFDIR)
      {
        _package_filename = path;
        checkPackage(path);
        std::cerr << "Package Filename " << path << std::endl;
      }
      else
      {
        _model_filename = path;
        checkModelfile(path);
        std::cerr << "Model Filename " << path << std::endl;
        _use_single_model = true;
      }
    }
    else
    {
      std::cerr << "Cannot find: " << path << "\n";
      exit(1);
    }
  };

  auto process_output_sizes = [&](const std::string &output_sizes_json_str) {
    Json::Value root;
    Json::Reader reader;
    if (!reader.parse(output_sizes_json_str, root, false))
    {
      std::cerr << "Invalid JSON format for output_sizes \"" << output_sizes_json_str << "\"\n";
      exit(1);
    }

    auto arg_map = argArrayToMap(root);
    for (auto &pair : arg_map)
    {
      uint32_t key = pair.first;
      Json::Value &val_json = pair.second;
      if (!val_json.isUInt())
      {
        std::cerr << "All the values in `output_sizes` must be unsigned integers\n";
        exit(1);
      }
      uint32_t val = val_json.asUInt();
      _output_sizes[key] = val;
    }
  };

  // General options
  po::options_description general("General options", 100);

  // clang-format off
  general.add_options()
    ("help,h", "Print available options")
    ("version", "Print version and exit immediately")
    ("nnpackage", po::value<std::string>()->notifier(process_nnpackage), "NN Package file(directory) name")
    ("modelfile", po::value<std::string>()->notifier(process_modelfile), "NN Model filename")
    ("path", po::value<std::string>()->notifier(process_path), "NN Package or NN Modelfile path")
    ("data_length", po::value<int>()->default_value(32)->notifier([&](const auto &v) { _data_length = v; }), "Data length number (default: 32)")
    ("input_data", po::value<std::string>()->notifier(process_inputdata), "NN Model Input data")
    ("expected_data", po::value<std::string>()->notifier(process_expecteddata), "NN Model Expected data")
    ("epoch", po::value<int>()->default_value(5)->notifier([&](const auto &v) { _epoch = v; }), "Epoch number (default: 5)")
    ("batch_size", po::value<int>()->default_value(32)->notifier([&](const auto &v) { _batch_size = v; }), "Batch size (default: 32)")
    ("learning_rate", po::value<float>()->default_value(1.0e-4)->notifier([&](const auto &v) { _learning_rate = v; }), "Learning rate (default: 1.0e-4)")
    ("loss", po::value<std::string>()->default_value("mse")->notifier([&] (const auto &v) { _loss_function = v; }), "Loss function name (default: mse)")
    ("optimizer", po::value<std::string>()->default_value("sgd")->notifier([&] (const auto &v) { _optimizer = v; }), "Optimizer name (default: sgd)")
    ("verbose_level,v", po::value<int>()->default_value(0)->notifier([&](const auto &v) { _verbose_level = v; }),
         "Verbose level\n"
         "0: prints the only result. Messages btw run don't print\n"
         "1: prints result and message btw run\n"
         "2: prints all of messages to print\n")
    ("output_sizes", po::value<std::string>()->notifier(process_output_sizes),
        "The output buffer size in JSON 1D array\n"
        "If not given, the model's output sizes are used\n"
        "e.g. '[0, 40, 2, 80]' to set 0th tensor to 40 and 2nd tensor to 80.\n")
    ;
  // clang-format on

  _options.add(general);
  _positional.add("path", -1);
}

void Args::Parse(const int argc, char **argv)
{
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(_options).positional(_positional).run(),
            vm);

  if (vm.count("help"))
  {
    std::cout << "onert_train\n\n";
    std::cout << "Usage: " << argv[0] << " [model path] [<options>]\n\n";
    std::cout << _options;
    std::cout << "\n";

    exit(0);
  }

  if (vm.count("version"))
  {
    _print_version = true;
    return;
  }

  {
    auto conflicting_options = [&](const std::string &o1, const std::string &o2) {
      if ((vm.count(o1) && !vm[o1].defaulted()) && (vm.count(o2) && !vm[o2].defaulted()))
      {
        throw boost::program_options::error(std::string("Two options '") + o1 + "' and '" + o2 +
                                            "' cannot be given at once.");
      }
    };

    // Cannot use both single model file and nnpackage at once
    conflicting_options("modelfile", "nnpackage");

    // Require modelfile, nnpackage, or path
    if (!vm.count("modelfile") && !vm.count("nnpackage") && !vm.count("path"))
      throw boost::program_options::error(
        std::string("Require one of options modelfile, nnpackage, or path."));
  }

  try
  {
    po::notify(vm);
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "Bad cast error - " << e.what() << '\n';
    exit(1);
  }
}

} // end of namespace onert_train
