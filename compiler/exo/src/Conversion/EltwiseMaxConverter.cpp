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

#include "EltwiseMaxConverter.h"

#include "GraphBlock.h"
#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco/Service/ShapeInference.h>

namespace
{

class EltwiseMaxInputHandler : public exo::InputHandler<loco::EltwiseMax, locoex::TFLMaximum>
{
public:
  void handover(loco::EltwiseMax *origin, locoex::TFLMaximum *replacer) override
  {
    replacer->x(origin->lhs());
    replacer->y(origin->rhs());
  }

  std::vector<loco::Node *> getInputsToConvert(loco::EltwiseMax *origin) override
  {
    std::vector<loco::Node *> inputs({origin->lhs(), origin->rhs()});
    return inputs;
  }

  void set(locoex::TFLMaximum *replacer, std::vector<loco::Node *> &to) override
  {
    assert(to.size() == 2);

    replacer->x(to.at(0));
    replacer->y(to.at(1));
  }

  void nullify(loco::EltwiseMax *origin) override
  {
    assert(origin);
    origin->lhs(nullptr);
    origin->rhs(nullptr);
  }
};

} // namespace

namespace exo
{

bool EltwiseMaxConverter::convert(loco::EltwiseMax *origin)
{
  EltwiseMaxInputHandler input_handler;
  exo::DomainConverter<loco::EltwiseMax, locoex::TFLMaximum> domain_converter;

  auto tfl_new = domain_converter.convert<FeatureLayout::NHWC>(origin, input_handler);

  return (tfl_new != nullptr);
}

} // namespace exo
