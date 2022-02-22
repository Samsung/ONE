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

#include "UserData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"
#include "Validation.h"

#include <cassert>
#include <stdexcept>

namespace
{

using namespace locomotiv;

void execute_node(loco::Pull *pull)
{
  auto input_data = user_data(pull);

  validate(input_data, "Input not ready");
  // User always passes a "Tensor"

  std::unique_ptr<NodeData> pull_data = nullptr;

  // Q. Is it possible to use generic one?
  switch (input_data->dtype())
  {
    case loco::DataType::S32:
    {
      auto input_bufptr = input_data->as_s32_bufptr();
      pull_data = make_data(*input_bufptr);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      pull_data = make_data(*input_bufptr);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(pull_data != nullptr);
  annot_data(pull, std::move(pull_data));
  annot_domain(pull, loco::Domain::Tensor);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::Pull *pull) { execute_node(pull); }

} // namespace locomotiv
