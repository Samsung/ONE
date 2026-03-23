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

#ifndef __KNOB_H__
#define __KNOB_H__

namespace moco
{
namespace tf
{

enum class Knob
{
#define KNOB_BOOL(NAME, DEFAULT, DESC) NAME,
#include "Knob.lst"
#undef KNOB_BOOL
};

template <Knob K> struct KnobTrait;

#define KNOB_BOOL(NAME, DEFAULT, DESC)     \
  template <> struct KnobTrait<Knob::NAME> \
  {                                        \
    using ValueType = bool;                \
  };
#include "Knob.lst"
#undef KNOB_BOOL

template <Knob K> typename KnobTrait<K>::ValueType get(void);

} // namespace tf
} // namespace moco

#endif // __KNOB_H__
