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

void execute_node(loco::Push *push)
{
  auto from_data = annot_data(push->from());

  validate(from_data, "Ingredient not ready");
  validate(annot_domain(push->from()) == loco::Domain::Tensor, "Ingredient of Push is not tensor");

  std::unique_ptr<NodeData> push_data = nullptr;

  switch (from_data->dtype())
  {
    case loco::DataType::S32:
    {
      auto from_bufptr = from_data->as_s32_bufptr();
      push_data = make_data(*from_bufptr);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      auto from_bufptr = from_data->as_f32_bufptr();
      push_data = make_data(*from_bufptr);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(push_data != nullptr);
  annot_data(push, std::move(push_data));
  annot_domain(push, loco::Domain::Tensor);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::Push *push) { execute_node(push); }

} // namespace locomotiv
