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

#include <iostream>
#include <vector>

#include "support/CommandLine.h"
#include "pass/PassException.h"
#include "Driver.h"

using namespace nnc;

/*
 * Prints the explanatory string of an exception. If the exception is nested, recurses to print
 * the explanatory string of the exception it holds.
 */
static void printException(const std::exception &e, int indent = 0)
{
  std::cerr << std::string(indent, ' ') << e.what() << std::endl;
  try
  {
    std::rethrow_if_nested(e);
  }
  catch (const std::exception &e)
  {
    printException(e, indent + 2);
  }
}

int main(int argc, const char *argv[])
{
  int exit_code = EXIT_FAILURE;

  try
  {
    // Parse command line
    cli::CommandLine::getParser()->parseCommandLine(argc, argv);

    //
    // run compiler pipeline:
    //
    // for_each(all_passes):
    //   run pass
    //
    Driver driver;
    driver.runDriver();

    // errors didn't happen
    exit_code = EXIT_SUCCESS;
  }
  catch (const DriverException &e)
  {
    printException(e);
    std::cerr << "use --help for more information" << std::endl;
  }
  catch (const PassException &e)
  {
    printException(e);
  }

  return exit_code;
}
