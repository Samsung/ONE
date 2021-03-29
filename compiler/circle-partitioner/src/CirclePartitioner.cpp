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

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>

namespace
{

const char *opt_prt = "--partition";
const char *opt_bks = "--backends";
const char *opt_def = "--default";

void print_version(void)
{
  std::cout << "circle-partitioner version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

void build_arser(arser::Arser &arser)
{
  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  arser.add_argument(opt_prt)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Partition information file which provides backend to assign");

  arser.add_argument(opt_bks)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Backends in CSV to use for partitioning");

  arser.add_argument(opt_def)
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Default backend to assign");

  arser.add_argument("input")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Input circle model file path");
  arser.add_argument("output")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Output parition folder path");
}

} // namespace

int entry(int argc, char **argv)
{
  arser::Arser arser("circle-partitioner provides circle model partitioning");

  build_arser(arser);

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cerr << arser;
    return 255;
  }

  return 0;
}
