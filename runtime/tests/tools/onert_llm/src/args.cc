/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "nnfw_util.h"

#include <functional>
#include <unistd.h>
#include <sys/stat.h>

namespace onert_llm
{

Args::Args(const int argc, char **argv)
{
  Initialize();
  Parse(argc, argv);
}

void Args::Initialize(void)
{
  _arser.add_argument("path").type(arser::DataType::STR).help("nnpackage path");

  arser::Helper::add_version(_arser, print_version);
  _arser.add_argument("--dump:raw").type(arser::DataType::STR).help("Raw output filename");
  _arser.add_argument("--load:raw").type(arser::DataType::STR).help("Raw input filename");
}

void Args::Parse(const int argc, char **argv)
{
  try
  {
    _arser.parse(argc, argv);

    if (_arser.get<bool>("--version"))
    {
      _print_version = true;
      return;
    }

    if (_arser["path"])
    {
      auto path = _arser.get<std::string>("path");
      struct stat sb;
      if (stat(path.c_str(), &sb) == 0)
      {
        if (sb.st_mode & S_IFDIR)
        {
          _package_filename = path;
          std::cout << "Package Filename " << path << std::endl;
        }
      }
      else
      {
        std::cerr << "Cannot find: " << path << "\n";
        exit(1);
      }
    }

    if (_arser["--dump:raw"])
      _dump_raw_filename = _arser.get<std::string>("--dump:raw");

    if (_arser["--load:raw"])
      _load_raw_filename = _arser.get<std::string>("--load:raw");
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "Bad cast error - " << e.what() << '\n';
    exit(1);
  }
}

} // end of namespace onert_llm
