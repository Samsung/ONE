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

/**
 * dummy-infer only tests its interface rather than its functionality.
 *
 * ./dummy-infer ${INPUT_NAME}
 * dummy-infer dummy output!!!
 */

#include <iostream>

int main(int argc, char **argv)
{
  if (argc != 2)
    return EXIT_FAILURE;

  std::cout << "dummy-infer dummy output!!!" << std::endl;

  return EXIT_SUCCESS;
}
