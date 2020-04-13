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

#include <direct.h>
#include <windows.h>

namespace filesystem
{

const std::string separator() { return "\\"; }

std::string normalize_path(const std::string &path)
{
  std::string ret = path;

  std::string candidate = "/";
  size_t start_pos = 0;
  while ((start_pos = ret.find(candidate, start_pos)) != std::string::npos)
  {
    ret.replace(start_pos, candidate.length(), separator());
    start_pos += separator().length();
  }
  return ret;
}

bool is_dir(const std::string &path)
{
  DWORD ftyp = GetFileAttributesA(path.c_str());
  if (ftyp == INVALID_FILE_ATTRIBUTES)
    return false; // something is wrong with path

  if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
    return true; // this is a directory

  return false; // this is not a directory
}

bool mkdir(const std::string &path) { return _mkdir(path.c_str()) == 0; }

} // namespace filesystem
