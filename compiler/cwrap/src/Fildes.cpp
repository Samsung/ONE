/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "cwrap/Fildes.h"

#include <cassert>
#include <unistd.h>

namespace
{

/**
 * @note  making inline to this function will prevent unused function error
 *        as error_value() is used only inside assert()
 */
inline bool error_value(int fd) { return fd == -1; }

inline bool valid_value(int fd) { return fd >= 0; }

} // namespace

namespace cwrap
{

Fildes::Fildes() : _value{-1}
{
  // DO NOTHING
}

Fildes::Fildes(int value) : _value{value}
{
  // DO NOTHING
  assert(error_value(value) || valid_value(value));
}

Fildes::Fildes(Fildes &&fildes)
{
  set(fildes.release());
  assert(error_value(fildes.get()));
}

Fildes::~Fildes()
{
  assert(error_value(_value) || valid_value(_value));

  if (valid_value(_value))
  {
    close(_value);
    _value = -1;
  }

  assert(error_value(_value));
}

Fildes &Fildes::operator=(Fildes &&fildes)
{
  set(fildes.release());
  return (*this);
}

int Fildes::get(void) const { return _value; }

void Fildes::set(int value)
{
  assert(error_value(_value) || valid_value(_value));

  if (valid_value(_value))
  {
    close(_value);
    _value = -1;
  }
  assert(error_value(_value));

  _value = value;
  assert(_value == value);
}

int Fildes::release(void)
{
  int res = get();
  _value = -1;
  return res;
}

bool valid(const Fildes &fildes) { return valid_value(fildes.get()); }

} // namespace cwrap
