/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Support.hpp"

#include <memory>
#include <cassert>
#include <fstream>
#include <stdexcept>

namespace
{

template <typename T>
std::unique_ptr<T> open_fstream(const std::string &path, std::ios_base::openmode mode)
{
  if (path == "-")
  {
    return nullptr;
  }

  auto stream = std::make_unique<T>(path.c_str(), mode);
  if (!stream->is_open())
  {
    throw std::runtime_error{"ERROR: Failed to open " + path};
  }
  return stream;
}

} // namespace

std::string Cmdline::get(unsigned int index) const
{
  if (index >= _argc)
    throw std::runtime_error("Argument index out of bound");

  return std::string(_argv[index]);
}

std::string Cmdline::get_or(unsigned int index, const std::string &s) const
{
  if (index >= _argc)
    return s;

  return std::string(_argv[index]);
}

std::unique_ptr<UI> make_ui(const Cmdline &cmdargs)
{
  auto iocfg = std::make_unique<UI>();

  auto in = open_fstream<std::ifstream>(cmdargs.get_or(0, "-"), std::ios::in | std::ios::binary);
  iocfg->in(std::move(in));

  auto out = open_fstream<std::ofstream>(cmdargs.get_or(1, "-"), std::ios::out | std::ios::binary);
  iocfg->out(std::move(out));

  return iocfg;
}
