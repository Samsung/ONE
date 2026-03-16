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
#include "RawModelLoader.h"

#include <cmdline/View.h>

#include <memory>
#include <fstream>
#include <cassert>

using std::make_unique;

extern "C" std::unique_ptr<enco::Frontend> make_frontend(const cmdline::View &cmdline)
{
  assert(cmdline.size() == 1); // tflite file name

  auto model = load_from(cmdline.at(0));

  return make_unique<Frontend>(std::move(model));
}
