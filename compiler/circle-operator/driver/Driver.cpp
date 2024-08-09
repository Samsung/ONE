/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Dump.h"

#include <arser/arser.h>
#include <foder/FileLoader.h>
#include <fstream>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <string>

#include <signal.h>

void handle_segfault(int signal, siginfo_t *si, void *arg)
{
  std::cerr << "ERROR: Failed to load file" << std::endl;
  exit(255);
}

int entry(int argc, char **argv)
{
  // TODO add option to dump for all sub-graphs
  arser::Arser arser{
    "circle-operator allows users to retrieve operator information from a Circle model file"};
  arser.add_argument("--name").nargs(0).help("Dump operators name in circle file");
  arser.add_argument("--code").nargs(0).help("Dump operators code in circle file");
  arser.add_argument("--shapes").nargs(0).help("Dump shapes");
  arser.add_argument("--output_path").help("Save output to file (default output is console)");
  arser.add_argument("circle").help("Circle file to dump");

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

  cirops::DumpOption option;
  option.names = arser["--name"];
  option.codes = arser["--code"];
  option.shapes = arser["--shapes"];

  std::ofstream oFstream;
  std::ostream *oStream = &std::cout;
  if (arser["--output_path"])
  {
    auto output_path = arser.get<std::string>("--output_path");
    oFstream.open(output_path, std::ofstream::out | std::ofstream::trunc);
    if (oFstream.fail())
    {
      std::cerr << "ERROR: Failed to create output to file " << output_path << std::endl;
      return 255;
    }
    oStream = &oFstream;
  }

  // hook segment fault
  struct sigaction sa;
  memset(&sa, 0, sizeof(struct sigaction));
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = handle_segfault;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, NULL);

  std::string modelFile = arser.get<std::string>("circle");
  // Load Circle model from a circle file
  try
  {
    foder::FileLoader fileLoader{modelFile};
    std::vector<char> modelData = fileLoader.load();
    const circle::Model *circleModel = circle::GetModel(modelData.data());
    if (circleModel == nullptr)
    {
      std::cerr << "ERROR: Failed to load circle '" << modelFile << "'" << std::endl;
      return 255;
    }
    cirops::DumpOperators dump;
    dump.run(*oStream, circleModel, option);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << "ERROR: " << err.what() << std::endl;
    return 255;
  }

  if (oFstream.is_open())
  {
    oFstream.close();
  }

  return 0;
}
