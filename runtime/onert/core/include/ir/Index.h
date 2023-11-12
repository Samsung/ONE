/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_INDEX_H__
#define __ONERT_IR_INDEX_H__

#include "util/Index.h"

#include <ostream>

namespace onert
{
namespace ir
{

struct OperationIndexTag;
using OperationIndex = ::onert::util::Index<uint32_t, OperationIndexTag>;

struct OperandIndexTag;
using OperandIndex = ::onert::util::Index<uint32_t, OperandIndexTag>;

struct IOIndexTag;
using IOIndex = ::onert::util::Index<uint32_t, IOIndexTag>;

struct SubgraphIndexTag;
using SubgraphIndex = ::onert::util::Index<uint16_t, SubgraphIndexTag>;

struct ModelIndexTag;
using ModelIndex = ::onert::util::Index<uint16_t, ModelIndexTag>;

struct OriginIndexTag;
using OriginIndex = ::onert::util::Index<uint32_t, OriginIndexTag>;

template <typename IndexType>
std::ostream &_index_print_impl(std::ostream &o, const std::string &prefix, IndexType index)
{
  if (index.undefined())
    return o << prefix << std::string("?");
  else
    return o << prefix << index.value();
}

inline std::ostream &operator<<(std::ostream &o, const OperationIndex &i)
{
  return _index_print_impl(o, "@", i);
}

inline std::ostream &operator<<(std::ostream &o, const OperandIndex &i)
{
  return _index_print_impl(o, "%", i);
}

inline std::ostream &operator<<(std::ostream &o, const IOIndex &i)
{
  return _index_print_impl(o, "IO", i);
}

inline std::ostream &operator<<(std::ostream &o, const SubgraphIndex &i)
{
  return _index_print_impl(o, "SUBGRAPH", i);
}

inline std::ostream &operator<<(std::ostream &o, const ModelIndex &i)
{
  return _index_print_impl(o, "MODEL", i);
}

inline std::ostream &operator<<(std::ostream &o, const OriginIndex &i)
{
  return _index_print_impl(o, "", i);
}
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_INDEX_H__
