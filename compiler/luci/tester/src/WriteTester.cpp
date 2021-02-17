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

#include "ReadModule.h"

#include <luci/CircleExporter.h>
#include <oops/InternalExn.h>

#include <fstream>
#include <iostream>
#include <string>

namespace
{

void show_help_message(const char *progname, std::ostream &os)
{
  os << "USAGE: " << progname << " circlefile_in circlefile_out" << std::endl << std::endl;
}

void show_error_message(const char *progname, std::ostream &os, const std::string &msg)
{
  os << "ERROR: " << msg << std::endl;
  os << std::endl;

  show_help_message(progname, os);
}

struct CircleExpContract : public luci::CircleExporter::Contract
{
public:
  CircleExpContract(loco::Graph *graph, const std::string &filename)
    : _graph(graph), _filepath(filename)
  {
    // NOTHING TO DO
  }
  CircleExpContract(luci::Module *module, const std::string &filename)
    : _module(module), _filepath(filename)
  {
    // NOTHING TO DO
  }
  virtual ~CircleExpContract() = default;

public:
  loco::Graph *graph(void) const final { return _graph; }

  luci::Module *module(void) const final { return _module; }

public:
  bool store(const char *ptr, const size_t size) const final;

private:
  loco::Graph *_graph{nullptr};
  luci::Module *_module{nullptr};
  const std::string _filepath;
};

bool CircleExpContract::store(const char *ptr, const size_t size) const
{
  if (!ptr)
    INTERNAL_EXN("Graph was not serialized by FlatBuffer for some reason");

  std::ofstream fs(_filepath.c_str(), std::ofstream::binary);
  fs.write(ptr, size);

  return fs.good();
}

} // namespace

/*
 * @brief WriteTester main
 *
 *        Give two Circle file as an argument
 *
 *        This will use luci_import to read the first file and get loco graph
 *        With the graph, this will use luci_export to write to the second file
 *        Like ReadTester, LUCI_LOG=1 environment variable is available to dump the graph
 */
int entry(int argc, char **argv)
{
  if (argc != 3)
  {
    show_error_message(argv[0], std::cerr, "In/Out Circle file path is not specified");
    return 255;
  }

  std::string input_path = argv[1];
  std::string output_path = argv[2];

  std::cout << "[INFO] Circle from '" << input_path << "' to '" << output_path << "'" << std::endl;

  auto module = ReadModule(input_path);
  if (module == nullptr)
    return EXIT_FAILURE;

  // Export to output Circle file
  luci::CircleExporter exporter;

  CircleExpContract contract(module.get(), output_path);

  return exporter.invoke(&contract) ? 0 : 255;
}
