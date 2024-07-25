/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Server.h"
#include "util/Logging.h"

using namespace npud;

int main(int argc, const char *argv[])
{
  auto &server = core::Server::instance();

  VERBOSE(main) << "Starting npud\n";
  try
  {
    server.run();
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  VERBOSE(main) << "Finished npud\n";
  return 0;
}
