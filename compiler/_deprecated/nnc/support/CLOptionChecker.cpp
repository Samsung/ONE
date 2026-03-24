/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "support/CommandLine.h"

#include <dirent.h>
#include <cstring>

namespace nnc
{
namespace cli
{

void checkInFile(const Option<std::string> &in_file)
{
  if (in_file.empty())
    throw BadOption("Input file name should not be empty");

  auto f = fopen(in_file.c_str(), "rb");
  if (!f)
    throw BadOption("Cannot open file <" + in_file + ">");
  fclose(f);
} // checkInFile

void checkOutFile(const Option<std::string> &out_file)
{
  if (out_file.empty())
    throw BadOption("Output file name should not be empty");

  /// @todo: if file already exists need to check accessibility

} // checkOutFile

void checkInDir(const Option<std::string> &dir)
{
  auto stream = opendir(dir.c_str());

  if (stream == nullptr)
    throw BadOption(std::string("Could not open directory: ") + std::strerror(errno) + ".");

  closedir(stream);
} // checkInDir

void checkOutDir(const Option<std::string> &dir)
{
  auto stream = opendir(dir.c_str());

  if (stream == nullptr)
  {
    // Do not consider the missing directory an error.
    if (errno == ENOENT)
      return;

    throw BadOption(std::string("Could not open directory: ") + std::strerror(errno) + ".");
  }

  closedir(stream);
} // checkOutDir

} // namespace cli
} // namespace nnc
