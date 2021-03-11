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

void execute_node(loco::BiasEncode *bias_enc)
{
  auto input_data = annot_data(bias_enc->input());

  validate(input_data, "Input not ready");
  validate(annot_domain(bias_enc->input()) == loco::Domain::Tensor,
           "Input domain should be Tensor");
  validate(input_data->shape()->rank() == 1, "Input data rank must be 1");

  std::unique_ptr<NodeData> bias_enc_data = nullptr;

  switch (input_data->dtype())
  {
    case loco::DataType::S32:
    {
      auto input_bufptr = input_data->as_s32_bufptr();
      bias_enc_data = make_data(*input_bufptr);
      break;
    }
    case loco::DataType::FLOAT32:
    {
      auto input_bufptr = input_data->as_f32_bufptr();
      bias_enc_data = make_data(*input_bufptr);
      break;
    }
    default:
      throw std::runtime_error("NYI for this DataType");
  }

  assert(bias_enc_data != nullptr);
  annot_data(bias_enc, std::move(bias_enc_data));
  annot_domain(bias_enc, loco::Domain::Bias);
}

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::BiasEncode *bias_enc) { execute_node(bias_enc); }

} // namespace locomotiv
