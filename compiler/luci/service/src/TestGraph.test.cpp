/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TestGraph.h"

namespace luci
{
namespace test
{

void graph_input_shape(luci::CircleInput *input)
{
  auto index = input->index();
  auto inputs = input->graph()->inputs();

  for (uint32_t idx = 0; idx < inputs->size(); ++idx)
  {
    auto gi = inputs->at(idx);
    if (gi->index() == index)
    {
      auto input_shape = std::make_unique<loco::TensorShape>();

      input_shape->rank(input->rank());
      for (uint32_t r = 0; r < input->rank(); ++r)
        input_shape->dim(r) = loco::Dimension(input->dim(r));

      gi->shape(std::move(input_shape));
      break;
    }
  }
}

void graph_output_shape(luci::CircleOutput *output)
{
  auto index = output->index();
  auto outputs = output->graph()->outputs();

  for (uint32_t idx = 0; idx < outputs->size(); ++idx)
  {
    auto go = outputs->at(idx);
    if (go->index() == index)
    {
      auto output_shape = std::make_unique<loco::TensorShape>();

      output_shape->rank(output->rank());
      for (uint32_t r = 0; r < output->rank(); ++r)
        output_shape->dim(r) = loco::Dimension(output->dim(r));

      go->shape(std::move(output_shape));
      break;
    }
  }
}

void graph_input_dtype(luci::CircleInput *input)
{
  auto index = input->index();
  auto inputs = input->graph()->inputs();

  for (uint32_t idx = 0; idx < inputs->size(); ++idx)
  {
    auto gi = inputs->at(idx);
    if (gi->index() == index)
    {
      gi->dtype(input->dtype());
      break;
    }
  }
}

void graph_output_dtype(luci::CircleOutput *output)
{
  auto index = output->index();
  auto outputs = output->graph()->outputs();

  for (uint32_t idx = 0; idx < outputs->size(); ++idx)
  {
    auto go = outputs->at(idx);
    if (go->index() == index)
    {
      go->dtype(output->dtype());
      break;
    }
  }
}

} // namespace test
} // namespace luci
