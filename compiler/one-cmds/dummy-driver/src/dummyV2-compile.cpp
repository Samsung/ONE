/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * dummyV2-compile only tests its interface rather than its functionality.
 *
 * ./dummyV2-compile -o ${OUTPUT_NAME} ${INPUT_NAME}
 *
 * NOTE argv[3](INPUT_NAME) is not used here.
 */

#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char **argv)
{
  if (argc != 4)
    return EXIT_FAILURE;

  std::string opt_o{"-O"};
  std::string argv_1{argv[1]};

  if (opt_o != argv_1)
  {
    std::cout << "dummyV2-compile: Invalid option" << std::endl;
    return EXIT_FAILURE;
  }

  std::string output_name{argv[2]};
  std::ofstream outfile(output_name);

  outfile << "dummyV2-compile dummy output!!" << std::endl;

  outfile.close();

  return EXIT_SUCCESS;
}
