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

#include "env.h"

#include <string>

#include <cstdlib>
#include <cassert>

inline void ensure(int err) { assert(err == 0); }

int main(int argc, char **argv)
{
  const std::string key{"TEST"};
  const int num{3};

  const auto str = std::to_string(num);

  ensure(unsetenv(key.c_str()));
  ensure(setenv(key.c_str(), str.c_str(), 0));

  int value = 0;

  assert(value != num);

  IntVar buffer(key, value);

  assert(buffer() == num);

  return 0;
}
