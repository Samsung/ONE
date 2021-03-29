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

#include "HelperPath.h"

#include <cassert>
#include <sstream>
#include <stdlib.h>

namespace partee
{

bool make_dir(const std::string &path)
{
  std::string command("mkdir -p ");
  command += path;
  int ret = ::system(command.c_str());
  return ret == 0;
}

std::string get_filename_ext(const std::string &base)
{
  // find last '/' to get filename.ext
  auto pos = base.find_last_of("/");
  if (pos == std::string::npos)
    return base;

  return base.substr(pos + 1);
}

std::string make_path(const std::string &base, const std::string &input, uint32_t idx,
                      const std::string &backend)
{
  auto filename_ext = get_filename_ext(input);

  // We will assume file type .circle if not given
  // TODO maybe throw if there is no extension?
  std::string filename = filename_ext;
  std::string ext = "circle";

  auto pos = filename_ext.find_last_of(".");
  if (pos != std::string::npos)
  {
    filename = filename_ext.substr(0, pos);
    ext = filename_ext.substr(pos + 1);
  }

  // format idx with 5 '0' paddings like '00123'
  uint32_t length = 5;
  auto seq = std::string(length, '0').append(std::to_string(idx));
  auto seq_fmt = seq.substr(seq.size() - length);

  return base + "/" + filename + "." + seq_fmt + "_" + backend + "." + ext;
}

} // namespace partee
