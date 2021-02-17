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

#include <iostream>
#include <string>

namespace
{

void show_help_message(const char *progname, std::ostream &os)
{
  os << "USAGE: " << progname << " circlefile" << std::endl << std::endl;
}

void show_error_message(const char *progname, std::ostream &os, const std::string &msg)
{
  os << "ERROR: " << msg << std::endl;
  os << std::endl;

  show_help_message(progname, os);
}

} // namespace

/*
 * @brief ReadTest main
 *
 *        Give one Circle file as an argument
 *
 *        This will use luci_import to read the file and get loco graph
 *        In luci_import, LUCI_LOG environment will be checked and will
 *        dump graph to console if set.
 *        i.e. "LUCI_LOG=1 luci_readtester mymodel.circle"
 */
int entry(int argc, char **argv)
{
  if (argc != 2)
  {
    show_error_message(argv[0], std::cerr, "Circle file is not specified");
    return 255;
  }

  std::string input_path = argv[1];

  std::cout << "[INFO] Circle is '" << input_path << "'" << std::endl;

  auto module = ReadModule(input_path);
  if (module == nullptr)
    return EXIT_FAILURE;

  return 0;
}
