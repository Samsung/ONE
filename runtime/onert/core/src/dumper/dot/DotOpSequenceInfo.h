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

#ifndef __ONERT_CORE_DUMPER_DOT_DOT_SUBGRAPH_INFO_H__
#define __ONERT_CORE_DUMPER_DOT_DOT_SUBGRAPH_INFO_H__

#include <unordered_set>

#include "ir/Index.h"
#include "ir/OpSequence.h"
#include "util/Set.h"

namespace onert
{
namespace dumper
{
namespace dot
{

class DotOpSequenceInfo
{
public:
  DotOpSequenceInfo(const ir::OpSequenceIndex &index, const ir::OpSequence &op_seq,
                    const util::Set<ir::OperandIndex> &shown_operands);

  ir::OpSequenceIndex index() const { return _index; }
  std::string label() const { return _label; }
  void label(const std::string &val) { _label = val; }
  std::string fillcolor() const { return _fillcolor; }
  void fillcolor(const std::string &val) { _fillcolor = val; }
  const std::unordered_set<ir::OperationIndex> &operations() const { return _operations; }
  const std::unordered_set<ir::OperandIndex> &operands() const { return _operands; }

private:
  ir::OpSequenceIndex _index;
  std::string _label;
  std::string _fillcolor;
  std::unordered_set<ir::OperationIndex> _operations;
  std::unordered_set<ir::OperandIndex> _operands;
};

} // namespace dot
} // namespace dumper
} // namespace onert

#endif // __ONERT_CORE_DUMPER_DOT_DOT_SUBGRAPH_INFO_H__
