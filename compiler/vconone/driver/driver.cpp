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

#include <vconone/vconone.h>

#include <string>
#include <iostream>

int main(int argc, char *argv[])
{
  auto str = vconone::get_string();
  if (argc >= 2)
  {
    for (int c = 1; c < argc; ++c)
      std::cout << argv[c] << " ";
    std::cout << "version " << str << std::endl;
    std::cout << vconone::get_copyright() << std::endl;
  }
  else
    std::cout << str;

  return 0;
}
