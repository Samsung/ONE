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
#include "nnfw.h"

#include <iostream>
#include <string>
#include <sys/stat.h>

#define NNPR_ENSURE_STATUS(a)        \
  do                                 \
  {                                  \
    if ((a) != NNFW_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

static void print_version()
{
  uint32_t version;
  NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
  std::cout << "ggma_run (nnfw runtime: v" << (version >> 24) << "."
            << ((version & 0x0000FF00) >> 8) << "." << (version & 0xFF) << ")" << std::endl;
}

namespace ggma_run
{

Args::Args(const int argc, char **argv)
{
  initialize();
  parse(argc, argv);
}

void Args::initialize(void)
{
  _arser.add_argument("path").type(arser::DataType::STR).help("nnpackage path");
  arser::Helper::add_version(_arser, print_version);
}

void Args::parse(const int argc, char **argv)
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
          _package_path = path;
          std::cout << "Package Filename " << path << std::endl;
        }
      }
      else
      {
        std::cerr << "Cannot find: " << path << "\n";
        exit(1);
      }
    }
  }
  catch (const std::bad_cast &e)
  {
    std::cerr << "Bad cast error - " << e.what() << '\n';
    exit(1);
  }
}

} // end of namespace ggma_run
