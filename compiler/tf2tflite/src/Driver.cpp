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

#include "CustomopConfLoader.h"

#include <moco/LoggingContext.h>
#include <moco/tf/Frontend.h>
#include <exo/LoggingContext.h>
#include <exo/TFLExporter.h>

#include <nnkit/support/tftestinfo/TensorInfoParser.h>

#include <locop/FormattedGraph.h>

#include <hermes/ConsoleReporter.h>
#include <hermes/EnvConfig.h>

#include <cassert>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

std::unique_ptr<loco::Graph> import(const moco::ModelSignature &sig, const std::string &path)
{
  moco::tf::Frontend frontend;
  return frontend.load(sig, path.c_str(), moco::tf::Frontend::FileType::Binary);
}

} // namespace

//
// Logging Support
//
namespace
{

struct Logger final : public hermes::Source
{
  Logger(hermes::Context *ctx) { activate(ctx->sources(), ctx->bus()); }
  ~Logger() { deactivate(); }
};

struct LoggingContext
{
  static hermes::Context *get(void)
  {
    using EnvConfig = hermes::EnvConfig<hermes::EnvFormat::BooleanNumber>;

    static hermes::Context *ctx = nullptr;

    if (ctx == nullptr)
    {
      ctx = new hermes::Context;
      ctx->sinks()->append(std::make_unique<hermes::ConsoleReporter>());
      ctx->config(std::make_unique<EnvConfig>("TF2TFLITE_Log"));
    }

    return ctx;
  }
};

void print_help()
{
  std::cerr << "Usage: tf2tflite <path/to/info> <path/to/pb> <path/to/tflite/model> " << std::endl
            << "Options: --customop <path/to/customop.conf>" << std::endl;
}

} // namespace

#define LOGGER(name) \
  ::Logger name { ::LoggingContext::get() }

#define INFO(name) HERMES_INFO(name)

int main(int argc, char **argv)
{
  using EnvConfig = hermes::EnvConfig<hermes::EnvFormat::BooleanNumber>;

  // This line allows users to control all the moco-tf loggers via TF2TFLITE_Log_Frontend
  moco::LoggingContext::get()->config(std::make_unique<EnvConfig>("TF2TFLITE_Log_Frontend"));
  // This line allows users to control all the exo-tflite loggers via TF2TFLITE_Log_Backend
  exo::LoggingContext::get()->config(std::make_unique<EnvConfig>("TF2TFLITE_Log_Backend"));

  LOGGER(l);

  // TODO We need better args parsing in future
  if (!(argc == 4 or argc == 6))
  {
    print_help();
    return 255;
  }

  std::string info_path{argv[1]};
  std::string tf_path{argv[2]}; // .pb file
  std::string tflite_path{argv[3]};

  std::cout << "Read '" << info_path << "'" << std::endl;

  moco::ModelSignature sig;
  {
    for (const auto &info : nnkit::support::tftestinfo::parse(info_path.c_str()))
    {
      switch (info->kind())
      {
        case nnkit::support::tftestinfo::ParsedTensor::Kind::Input:
          sig.add_input(moco::TensorName{info->name()});
          sig.shape(info->name(), info->shape());
          break;

        case nnkit::support::tftestinfo::ParsedTensor::Kind::Output:
          sig.add_output(moco::TensorName{info->name()});
          sig.shape(info->name(), info->shape());
          break;

        default:
          throw std::runtime_error{"Unknown kind"};
      }
    }
  }

  if (argc == 6) // optional parameter: path of customop.conf
  {
    if (std::string{argv[4]} == "--customop")
    {
      tf2tflite::load_customop_conf(argv[5], sig);
    }
    else
    {
      print_help();
      return 255;
    }
  }

  std::cout << "Read '" << info_path << "' - Done" << std::endl;

  std::cout << "Import from '" << tf_path << "'" << std::endl;
  auto g = import(sig, tf_path);
  std::cout << "Import from '" << tf_path << "' - Done" << std::endl;

  INFO(l) << "Import Graph" << std::endl;
  INFO(l) << locop::fmt<locop::Formatter::LinearV1>(g) << std::endl;

  std::cout << "Export into '" << tflite_path << "'" << std::endl;
  exo::TFLExporter(g.get()).dumpToFile(tflite_path.c_str());
  std::cout << "Export into '" << tflite_path << "' - Done" << std::endl;

  return 0;
}
