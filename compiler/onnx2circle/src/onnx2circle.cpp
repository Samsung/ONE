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

#include "exo/LoggingContext.h"
#include "exo/CircleExporter.h"

#include "mir2loco.h"
#include "ONNXImporterImpl.h"

#include "locop/FormattedGraph.h"

#include "hermes/ConsoleReporter.h"
#include "hermes/EnvConfig.h"

#include <cassert>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <string>

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
      ctx->config(std::make_unique<EnvConfig>("ONNX2CIRCLE_Log"));
    }

    return ctx;
  }
};

void print_help()
{
  std::cerr << "Usage: onnx2circle <path/to/onnx> <path/to/circle/model> " << std::endl;
}

} // namespace

#define LOGGER(name) \
  ::Logger name { ::LoggingContext::get() }

#define INFO(name) HERMES_INFO(name)

int main(int argc, char **argv)
{
  using EnvConfig = hermes::EnvConfig<hermes::EnvFormat::BooleanNumber>;

  // This line allows users to control all the exo-circle loggers via ONNX2CIRCLE_Log_Backend
  exo::LoggingContext::get()->config(std::make_unique<EnvConfig>("ONNX2CIRCLE_Log_Backend"));

  LOGGER(l);

  // TODO We need better args parsing in future
  if (!(argc == 3))
  {
    print_help();
    return 255;
  }

  std::string onnx_path{argv[1]}; // .pb file
  std::string circle_path{argv[2]};

  std::cout << "Import from '" << onnx_path << "'" << std::endl;
  auto mir_g = mir_onnx::loadModel(onnx_path);
  auto loco_g = mir2loco::Transformer().transform(mir_g.get());
  std::cout << "Import from '" << onnx_path << "' - Done" << std::endl;

  INFO(l) << "Import Graph" << std::endl;
  INFO(l) << locop::fmt<locop::Formatter::LinearV1>(loco_g) << std::endl;

  std::cout << "Export into '" << circle_path << "'" << std::endl;
  exo::CircleExporter(loco_g.get()).dumpToFile(circle_path.c_str());
  std::cout << "Export into '" << circle_path << "' - Done" << std::endl;

  return 0;
}
