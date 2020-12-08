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

#include <algorithm>

#include "Linear.h"

#include "backend/IConfig.h"
#include "backend/Backend.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{

std::vector<ir::OpSequenceIndex> Linear::linearize(const compiler::LoweredGraph &lowered_graph)
{
  std::vector<ir::OpSequenceIndex> order;
  lowered_graph.iterateTopolOpSeqs(
      [&](const ir::OpSequenceIndex &index, const ir::OpSequence &) -> void {
        order.emplace_back(index);
      });
  return order;
}

void Linear::dump(const compiler::LoweredGraph &lowered_graph,
                  const std::vector<ir::OpSequenceIndex> &order)
{
  {
    const auto &toString = [](const onert::backend::Backend *backend) {
      assert(backend);
      std::string str;
      str += backend->config()->id();
      return "{" + str + "}";
    };

    VERBOSE(Linear) << "Final OpSequence" << std::endl;
    for (const auto index : order)
    {
      const auto &op_seq = lowered_graph.op_seqs().at(index);
      const auto lower_info = lowered_graph.getLowerInfo(index);
      const auto &operations = lowered_graph.graph().operations();
      VERBOSE(Linear) << "* OP_SEQ " << toString(lower_info->backend()) << " "
                      << ir::getStrFromOpSeq(op_seq, operations) << std::endl;
    }
  }
}

} // namespace compiler
} // namespace onert
