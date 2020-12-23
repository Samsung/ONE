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

#include "locomotiv/Session.h"
#include "locomotiv/NodeData.h"

#include "UserData.h"
#include "NodeDataImpl.h"
#include "NodeExecution.h"
#include "NodeDomain.h"

#include <cassert>

namespace locomotiv
{

Session::~Session()
{
  for (uint32_t i = 0; i < _graph->nodes()->size(); ++i)
  {
    auto node = _graph->nodes()->at(i);
    erase_user_data(node);
    erase_annot_data(node);
    erase_annot_domain(node);
  }
}

void Session::set_input(uint32_t index, std::unique_ptr<NodeData> &&data)
{
  assert(index < input_size());

  // Check whether already annotated
  auto pull = loco::pull_node(_graph, index);
  if (user_data(pull))
  {
    throw std::runtime_error("Graph input already has NodeData");
  }

  // Check data type match
  if (pull->dtype() != data->dtype())
  {
    throw std::runtime_error("Data type mismatch");
  }

  // Check shape match
  auto shape = data->shape();
  if (pull->rank() != shape->rank())
  {
    throw std::runtime_error("Shape rank mismatch");
  }
  for (uint32_t i = 0; i < pull->rank(); ++i)
  {
    if (pull->dim(i).known() && pull->dim(i).value() != shape->dim(i))
    {
      throw std::runtime_error("Shape dimension mismatch");
    }
  }

  user_data(pull, std::move(data));
}

void Session::infer()
{
  auto schedules = loco::postorder_traversal(_outputs);

  for (auto node : schedules)
  {
    NodeExecution::get().run(node);
  }
}

const NodeData *Session::get_output(uint32_t index)
{
  assert(index < output_size());

  auto output_node = _outputs.at(index);
  return annot_data(output_node);
}

} // namespace locomotiv
