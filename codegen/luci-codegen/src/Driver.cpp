/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Codegen.h"
#include "luci/Importer.h"
#include "luci/CircleExporter.h"
#include "luci/CircleFileExpContract.h"
#include "arser/arser.h"

#include <boost/filesystem.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

// string constants
static const std::string version = "0.1.0";

static const std::string input_path_arg = "input_path";
static const std::string output_path_arg = "output_path";
static const std::string version_arg = "--version";
static const std::string help_arg = "--help";
static const std::string inline_buffer_threshold_arg = "--inline_buffer_size";
static const std::string debug_arg = "--debug";
static const std::string architecture_arg = "--arch";
static const std::string cache_size_arg = "--cache_size";
static const std::string os_arg = "--os";
static const std::string scheduler_arg = "--scheduler";

struct CommandLineOptions
{
  std::string input_file_path;
  std::string output_package_path;
  luci_codegen::CodegenOptions codegen_options;
};

void print_version()
{
  std::cout << "version: " << version << "\n";
}

void print_help(const arser::Arser &cmd_options)
{
  std::cout << cmd_options << "\n";
}

void print_help_on_error_and_exit(const arser::Arser &cmp_options, std::string error)
{
  std::cerr << error << "\n" << cmp_options;
  exit(1);
}

CommandLineOptions parse_command_line(int argc, char **argv)
{
  CommandLineOptions opts;

  arser::Arser cmd_options("fuses parts of NN into custom operations in native code");

  cmd_options.add_argument(input_path_arg)
    .nargs(1)
    .required(true)
    .type(arser::DataType::STR)
    .help("Path to circle model file");

  cmd_options.add_argument(output_path_arg)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("path to directory where save output model and compiled operators");

  cmd_options.add_argument(version_arg)
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  cmd_options.add_argument(help_arg)
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show help information and exit")
    .exit_with([&cmd_options](){print_help(cmd_options);});

  cmd_options.add_argument(inline_buffer_threshold_arg)
    .nargs(1)
    .type(arser::DataType::INT32)
    .required(false)
    .help("Max size of constant buffers in bytes to inline into generated code");

  cmd_options.add_argument(debug_arg)
    .nargs(0)
    .required(false)
    .help("Generate code with correctness checks");

  cmd_options.add_argument(architecture_arg)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Target architecture for generated code. Possible options are: native, x86_32, x86_64, arm_32, arm_64");

  cmd_options.add_argument(cache_size_arg)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Target CPU cache size in bytes");

  cmd_options.add_argument(os_arg)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Target OS for generated code. Possible options are: native, linux, windows, android");

  cmd_options.add_argument(scheduler_arg)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Which scheduler to use. Possible options are: none, mullapudi, li, adams");

  try
  {
    cmd_options.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    print_help_on_error_and_exit(cmd_options, err.what());
  }

  opts.input_file_path = cmd_options.get<std::string>(input_path_arg);
  opts.output_package_path = cmd_options.get<std::string>(output_path_arg);

  if (cmd_options[inline_buffer_threshold_arg])
  {
    opts.codegen_options.max_inline_buffer_threshold = cmd_options.get<std::int32_t>(inline_buffer_threshold_arg);
  }

  opts.codegen_options.debug = cmd_options[debug_arg];

  if (cmd_options[cache_size_arg])
  {
    opts.codegen_options.arch.last_level_cache_size = cmd_options.get<std::int32_t>(cache_size_arg);
  }

  if (cmd_options[os_arg])
  {
    const std::unordered_map<std::string, luci_codegen::OS> os_translation = {{"native", luci_codegen::OS::Native}, {"android", luci_codegen::OS::Android}, {"linux", luci_codegen::OS::Linux}, {"windows", luci_codegen::OS::Windows}};
    auto target_os = cmd_options.get<std::string>(os_arg);
    if (os_translation.count(target_os) == 0)
    {
      print_help_on_error_and_exit(cmd_options, "Unknown target os: \"" + target_os + "\"");
    }
    opts.codegen_options.os = os_translation.at(target_os);
  }

  if (cmd_options[scheduler_arg])
  {
    const std::unordered_map<std::string, luci_codegen::SchedulerAlgorithm > scheduler_translation = {{"none", luci_codegen::SchedulerAlgorithm::None}, {"adams", luci_codegen::SchedulerAlgorithm::Adams}, {"mullapudi", luci_codegen::SchedulerAlgorithm::Mullapudi}, {"li", luci_codegen::SchedulerAlgorithm::Li}};
    auto scheduler = cmd_options.get<std::string>(scheduler_arg);
    if (scheduler_translation.count(scheduler) == 0)
    {
      print_help_on_error_and_exit(cmd_options, "Unknown scheduler: \"" + scheduler + "\"");
    }
    opts.codegen_options.scheduler = scheduler_translation.at(scheduler);
  }

  if (cmd_options[architecture_arg])
  {
    const std::unordered_map<std::string, luci_codegen::ArchType> architecture_translation = {{"native", luci_codegen::ArchType::Native}, {"arm_32", luci_codegen::ArchType::ARM_32}, {"arm_64", luci_codegen::ArchType::ARM_64}, {"x86_32", luci_codegen::ArchType::X86_32}, {"x86_64", luci_codegen::ArchType::X86_64}};
    auto target_architecture = cmd_options.get<std::string>(architecture_arg);
    if (architecture_translation.count(target_architecture) == 0)
    {
      print_help_on_error_and_exit(cmd_options, "Unknown target architecture: \"" + target_architecture + "\"");
    }
    opts.codegen_options.arch.type = architecture_translation.at(target_architecture);
  }

  return opts;
}

std::vector<char> read_file(const std::string &path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    std::cerr << "failed to open \"" << path << "\" file\n";
    exit(1);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (file.read(buffer.data(), size).bad())
  {
    std::cerr << "failed to read \"" << path << "\" file\n";
    exit(1);
  }
  return buffer;
}

int main(int argc, char **argv)
{

  auto options = parse_command_line(argc, argv);

  luci::Importer importer;

  auto raw_model_data = read_file(options.input_file_path);

  const circle::Model *circle_module = circle::GetModel(raw_model_data.data());
  std::unique_ptr<luci::Module> luci_module = importer.importModule(circle_module);

  luci_codegen::Codegen codegen(options.codegen_options);
  codegen.process_module(*luci_module);

  boost::filesystem::path output_package_dir(options.output_package_path);
  if (!boost::filesystem::exists(output_package_dir))
  {
    boost::filesystem::create_directory(output_package_dir);
  }
  boost::filesystem::path output_model_path = output_package_dir / "model.circle";
  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(luci_module.get(), output_model_path.string());
  exporter.invoke(&contract);

  codegen.emit_code(options.output_package_path);
  return 0;
}
