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

#include "Frontend.h"
#include "Importer.h"

#include <cmdline/View.h>

#include <memory>
#include <fstream>
#include <cassert>

extern "C" std::unique_ptr<enco::Frontend> make_frontend(const cmdline::View &cmdline)
{
  assert(cmdline.size() == 2);

  auto frontend = std::make_unique<Frontend>();

  // Fill prototxt
  {
    std::ifstream ifs{cmdline.at(0)};
    if (!ifs.is_open())
    {
      throw std::runtime_error("Prototxt file open fail");
    }

    if (!from_txt(ifs, *frontend->prototxt()))
    {
      throw std::runtime_error("Filling prototxt fail");
    }
  }

  // Fill caffemodel
  {
    std::ifstream ifs{cmdline.at(1), std::ios::binary};
    if (!ifs.is_open())
    {
      throw std::runtime_error("Caffemodel file open fail");
    }

    if (!from_bin(ifs, *frontend->caffemodel()))
    {
      throw std::runtime_error("Filling caffemodel fail");
    }
  }

  return std::move(frontend);
}
