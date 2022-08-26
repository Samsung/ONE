/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_UTIL_CLONABLE_H__
#define __ONERT_UTIL_CLONABLE_H__

#include <string>

namespace onert
{

/**
 * @brief Class to define interface for clonable classes
 */
template <typename Derived> struct Clonable
{
  virtual ~Clonable() = default;

  std::unique_ptr<Derived> clone() { return std::unique_ptr<Derived>(cloneImpl()); }

protected:
  virtual Derived *cloneImpl() const = 0;
};

} // namespace onert

#endif // __ONERT_UTIL_CLONABLE_H__
