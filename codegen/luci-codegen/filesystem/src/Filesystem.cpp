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

#include "Filesystem.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace luci_codegen_filesystem
{

path operator/(const path &lhs, const path &rhs)
{
  path p(lhs);
  p /= rhs;
  return p;
}

bool exists(const path &p)
{
  struct stat statbuf;
  int res = stat(p.c_str(), &statbuf);
  return res == 0;
}

bool is_directory(const path &p)
{
  struct stat statbuf;
  int res = stat(p.c_str(), &statbuf);
  if (res != 0)
  {
    return false;
  }
  return statbuf.st_mode & S_IFDIR;
}

bool create_directory(const path &p)
{
  mode_t mode = 0777;
  int res = mkdir(p.c_str(), mode);
  if (res != 0)
  {
    auto errn = errno;
    if (!is_directory(p))
    {
      throw filesystem_error(strerror(errn), p, std::error_code(errn, std::generic_category()));
    }
  }
  return res == 0;
}

} // namespace luci_codegen_filesystem
