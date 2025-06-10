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

#include "circleimpexp.h"

#include <arser/arser.h>

int safe_main(int argc, char *argv[])
{
  arser::Arser arser;

  // two positional arguments
  arser.add_argument("import").help("Input Circle file");
  arser.add_argument("export").help("Output Circle file");

  arser.parse(argc, argv);

  CirImpExpParam param;
  param.sourcefile = arser.get<std::string>("import");
  param.targetfile = arser.get<std::string>("export");

  return entry(param);
}

int main(int argc, char *argv[])
{
  try
  {
    return safe_main(argc, argv);
  }
  catch (const std::exception &err)
  {
    std::cout << err.what() << '\n';
  }

  return -1;
}
