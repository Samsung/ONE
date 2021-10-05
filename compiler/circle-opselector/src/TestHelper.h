/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_OPSELECTOR_TEST_HELPER_H__
#define __CIRCLE2CIRCLE_TEST_HELPER_H__

#include "ModuleIO.h"

#include <cassert>
#include <cstdio>
#include <vector>
#include <string.h>
#include <string>

template <size_t N> class Argv
{
public:
  typedef char *pchar_t;

public:
  ~Argv()
  {
    for (size_t n = 0; n < _ptr; ++n)
      delete _argv[n];
  }

  void add(const char *in)
  {
    assert(_ptr < N);
    _argv[_ptr] = new char[strlen(in) + 1];
    strncpy(_argv[_ptr], in, strlen(in) + 1);
    _ptr++;
  }

  pchar_t *argv(void) { return _argv; }

private:
  pchar_t _argv[N] = {
    nullptr,
  };
  size_t _ptr = 0;
};

#endif // __CIRCLE2CIRCLE_TEST_HELPER_H__
