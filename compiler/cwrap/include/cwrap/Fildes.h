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

#ifndef __CWRAP_FILDES_H__
#define __CWRAP_FILDES_H__

namespace cwrap
{

/**
 * @brief POSIX File Descriptor
 *
 * @note Fildes owns underlying file descriptor
 */
class Fildes final
{
public:
  Fildes();
  explicit Fildes(int value);

  // NOTE Copy is not allowed
  Fildes(const Fildes &) = delete;
  Fildes(Fildes &&);

  ~Fildes();

public:
  Fildes &operator=(Fildes &&);

public:
  int get(void) const;
  void set(int value);

  int release(void);

private:
  int _value{-1};
};

bool valid(const Fildes &);

} // namespace cwrap

#endif // __CWRAP_FILDES_H__
