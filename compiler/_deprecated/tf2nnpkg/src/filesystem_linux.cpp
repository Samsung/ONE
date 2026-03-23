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

#include <sys/stat.h>
#include <dirent.h>

namespace filesystem
{

const std::string separator() { return "/"; }

std::string normalize_path(const std::string &path)
{
  // DO NOTHING
  return path;
}

bool is_dir(const std::string &path)
{
  DIR *dir = opendir(path.c_str());
  if (dir)
  {
    closedir(dir);
    return true;
  }
  return false;
}

bool mkdir(const std::string &path) { return ::mkdir(path.c_str(), 0775) == 0; }

} // namespace filesystem
