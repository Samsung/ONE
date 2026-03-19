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

#include "filesystem.h"

#include <moco/LoggingContext.h>
#include <moco/tf/Frontend.h>
#include <exo/LoggingContext.h>
#include <exo/CircleExporter.h>

#include <nnkit/support/tftestinfo/TensorInfoParser.h>

#include <locop/FormattedGraph.h>

#include <hermes/ConsoleReporter.h>
#include <hermes/EnvConfig.h>

#include <memory>
#include <iostream>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

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
      ctx->config(std::make_unique<EnvConfig>("TF2NNPKG_Log"));
    }

    return ctx;
  }
};

void print_help()
{
  std::cerr << "Usage:" << std::endl;
  std::cerr << "    tf2nnpkg --info <path/to/info>" << std::endl;
  std::cerr << "             --graphdef <path/to/pb>" << std::endl;
  std::cerr << "             -o <path/to/package/dir>" << std::endl;
}

} // namespace

#define LOGGER(name) \
  ::Logger name { ::LoggingContext::get() }

#define INFO(name) HERMES_INFO(name)

namespace
{

void internal_error(void)
{
  std::cerr << "tf2nnpkg: internal compiler error" << std::endl;

  // TODO Explain how to report a bug
}

} // namespace

namespace
{

std::string extract_modelname(std::string tf_path)
{
  auto filename = filesystem::basename(tf_path);
  // TODO Find better way
  const std::string key = ".pb";
  auto suffix_index = filename.find(key);
  assert(suffix_index != std::string::npos);
  assert(suffix_index + key.size() == filename.size());

  return filename.substr(0, suffix_index);
}

class EntryFunctor
{
public:
  EntryFunctor();

public:
  ~EntryFunctor();

public:
  int operator()(int argc, char **argv) const;
};

EntryFunctor::EntryFunctor()
{
  // NOTE Implement initialization here
}

EntryFunctor::~EntryFunctor()
{
  // NOTE Implement finialization here
}

int EntryFunctor::operator()(int argc, char **argv) const
{
  using EnvConfig = hermes::EnvConfig<hermes::EnvFormat::BooleanNumber>;

  // This line allows users to control all the moco-tf loggers via TF2NNPKG_Log_Frontend
  moco::LoggingContext::get()->config(std::make_unique<EnvConfig>("TF2NNPKG_Log_Frontend"));
  // This line allows users to control all the exo-circle loggers via TF2NNPKG_Log_Backend
  exo::LoggingContext::get()->config(std::make_unique<EnvConfig>("TF2NNPKG_Log_Backend"));

  LOGGER(l);

  // Simple argument parser (based on map)
  std::map<std::string, std::function<void(const std::string &arg)>> argparse;

  std::string arg_info;
  std::string arg_graphdef;
  std::string arg_output;

  argparse["--info"] = [&](const std::string &arg) { arg_info = arg; };
  argparse["--graphdef"] = [&](const std::string &arg) { arg_graphdef = arg; };
  argparse["-o"] = [&](const std::string &arg) { arg_output = arg; };

  // TODO We need better args parsing in future

  for (int n = 1; n < argc; n += 2)
  {
    const std::string tag{argv[n]};
    const std::string arg{argv[n + 1]};

    auto it = argparse.find(tag);
    if (it == argparse.end())
    {
      std::cerr << "Option '" << tag << "' is not supported" << std::endl;
      print_help();
      return 255;
    }

    it->second(arg);
  }
  if (arg_info.empty() || arg_graphdef.empty() || arg_output.empty())
  {
    print_help();
    return 255;
  }

  // Input paths
  std::string info_path = arg_info;
  std::string tf_path = arg_graphdef; // .pb file

  // Output paths
  std::string outdir_path = arg_output;
  std::string modelname = extract_modelname(filesystem::normalize_path(tf_path));
  std::string nnpkg_path = filesystem::join(outdir_path, modelname);
  std::string model_filename = modelname + ".circle";
  std::string metadata_path = filesystem::join(nnpkg_path, "metadata");
  std::string circle_path = filesystem::join(nnpkg_path, model_filename);
  std::string manifest_path = filesystem::join(metadata_path, "MANIFEST");

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

  std::cout << "Read '" << info_path << "' - Done" << std::endl;

  std::cout << "Import from '" << tf_path << "'" << std::endl;
  auto g = import(sig, tf_path);
  std::cout << "Import from '" << tf_path << "' - Done" << std::endl;

  INFO(l) << "Import Graph" << std::endl;
  INFO(l) << locop::fmt<locop::Formatter::LinearV1>(g) << std::endl;

  if (not filesystem::is_dir(outdir_path))
  {
    std::cout << "Make output directory '" << outdir_path << "'" << std::endl;
    if (not filesystem::mkdir(outdir_path))
      throw std::runtime_error("Fail to make directory " + outdir_path);
    std::cout << "Make output directory '" << outdir_path << "' - Done" << std::endl;
  }

  if (not filesystem::is_dir(nnpkg_path))
  {
    std::cout << "Make package directory '" << nnpkg_path << "'" << std::endl;
    if (not filesystem::mkdir(nnpkg_path))
      throw std::runtime_error("Fail to make directory " + nnpkg_path);
    std::cout << "Make package directory '" << nnpkg_path << "' - Done" << std::endl;
  }

  std::cout << "Export into '" << circle_path << "'" << std::endl;
  exo::CircleExporter(g.get()).dumpToFile(circle_path.c_str());
  std::cout << "Export into '" << circle_path << "' - Done" << std::endl;

  if (not filesystem::is_dir(metadata_path))
  {
    std::cout << "Make metadata directory '" << metadata_path << "'" << std::endl;
    if (not filesystem::mkdir(metadata_path))
      throw std::runtime_error("Fail to make directory " + metadata_path);
    std::cout << "Make metadata directory '" << metadata_path << "' - Done" << std::endl;
  }

  std::cout << "Make manifest file '" << manifest_path << "'" << std::endl;
  std::ofstream manifest_file;
  manifest_file.open(manifest_path, std::ios::out | std::ios::binary);
  manifest_file << "{\n";
  manifest_file << "  \"major-version\" : \"1\",\n";
  manifest_file << "  \"minor-version\" : \"0\",\n";
  manifest_file << "  \"patch-version\" : \"0\",\n";
  manifest_file << "  \"models\" : [ \"" + model_filename + "\" ],\n";
  manifest_file << "  \"model-types\" : [ \"circle\" ]\n";
  manifest_file << "}";
  manifest_file.close();
  std::cout << "Make manifest file '" << manifest_path << "' - Done" << std::endl;

  return 0;
}

} // namespace

int main(int argc, char **argv)
{
  // TODO Add "signal" handler here

  try
  {
    EntryFunctor entry;
    return entry(argc, argv);
  }
  catch (...)
  {
    // Catch all the exception and print the default error message.
    internal_error();
  }

  // EX_SOFTWARE defined in "sysexits.h"
  return 70;
}
