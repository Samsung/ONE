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

#ifndef __LOCOP_INTERFACES_H__
#define __LOCOP_INTERFACES_H__

#include <ostream>

namespace locop
{

enum class Interface
{
  Formatted,
};

template <Interface I> struct Spec;

template <> struct Spec<Interface::Formatted>
{
  virtual ~Spec() = default;

  virtual void dump(std::ostream &os) const = 0;
};

std::ostream &operator<<(std::ostream &, const Spec<Interface::Formatted> &);

} // namespace locop

#endif // __LOCOP_INTERFACES_H__
