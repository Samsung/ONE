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

#include "EltwiseSqrtConverter.h"

#include "GraphBlock.h"
#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco/Service/ShapeInference.h>

namespace
{

class EltwiseSqrtInputHandler : public exo::InputHandler<loco::EltwiseSqrt, locoex::TFLSqrt>
{
public:
  void handover(loco::EltwiseSqrt *origin, locoex::TFLSqrt *replacer) override
  {
    replacer->x(origin->input());
  }

  std::vector<loco::Node *> getInputsToConvert(loco::EltwiseSqrt *origin) override
  {
    std::vector<loco::Node *> inputs({origin->input()});
    return inputs;
  }

  void set(locoex::TFLSqrt *replacer, std::vector<loco::Node *> &to) override
  {
    assert(to.size() == 1);

    replacer->x(to.at(0));
  }

  void nullify(loco::EltwiseSqrt *origin) override { origin->input(nullptr); }
};

} // namespace

namespace exo
{

bool EltwiseSqrtConverter::convert(loco::EltwiseSqrt *origin)
{
  EltwiseSqrtInputHandler input_handler;
  exo::DomainConverter<loco::EltwiseSqrt, locoex::TFLSqrt> domain_converter;

  auto tfl_new = domain_converter.convert<FeatureLayout::NHWC>(origin, input_handler);

  return (tfl_new != nullptr);
}

} // namespace exo
