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

/**
 * dummy-compile only tests its interface rather than its functionality.
 *
 * ./dummy-compile -o ${OUTPUT_NAME} ${INPUT_NAME}
 * ./dummy-compile -o ${OUTPUT_NAME} ${INPUT_NAME} -T {TARGET_NAME}
 *
 * NOTE argv[3](INPUT_NAME) is not used here.
 */

#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char **argv)
{
  if (argc != 4 and argc != 6)
    return EXIT_FAILURE;

  std::string opt_o{"-o"};
  std::string argv_1{argv[1]};

  if (opt_o != argv_1)
    return EXIT_FAILURE;

  std::string output_name{argv[2]};
  std::ofstream outfile(output_name);

  if (argc == 4)
  {
    outfile << "dummy-compile dummy output!!" << std::endl;
  }
  // argc == 6
  else
  {
    std::string opt_T{"--target"};
    std::string argv_4{argv[4]};
    if (opt_T != argv_4)
      return EXIT_FAILURE;

    std::string target_name(argv[5]);
    outfile << "dummy-compile with " << target_name << " target" << std::endl;
  }

  outfile.close();

  return EXIT_SUCCESS;
}
