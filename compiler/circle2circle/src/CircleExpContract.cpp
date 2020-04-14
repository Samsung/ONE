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

#include "CircleExpContract.h"

#include <oops/InternalExn.h>

#include <fstream>
#include <iostream>

bool CircleExpContract::store(const char *ptr, const size_t size) const
{
  if (!ptr)
    INTERNAL_EXN("Graph was not serialized by FlatBuffer for some reason");

  std::ofstream fs(_filepath.c_str(), std::ofstream::binary);
  fs.write(ptr, size);

  return fs.good();
}
