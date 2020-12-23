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

#include "NodeExecution.h"

#include "NodeDataImpl.h"
#include "NodeDomain.h"
#include "Validation.h"

#include <stdexcept>
#include <cassert>

namespace
{

using namespace locomotiv;

void execute_node(loco::Forward *forward)
{
  auto input_data = annot_data(forward->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(forward->input()) != loco::Domain::Unknown,
           "Input domain must not Unknown");

  std::unique_ptr<NodeData> forward_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::S32:
    {
      auto input_bufptr = input_data->as_s32_bufptr();
      forward_data = make_data(*input_bufptr);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      forward_data = make_data(*input_bufptr);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(forward_data != nullptr);
  annot_data(forward, std::move(forward_data));
  annot_domain(forward, annot_domain(forward->input()));
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::Forward *forward) { execute_node(forward); }

} // namespace locomotiv
