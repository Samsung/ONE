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

#include "Relu6Converter.h"

#include "GraphBlock.h"
#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco/Service/ShapeInference.h>

namespace
{

class Relu6InputHandler : public exo::InputHandler<loco::ReLU6, locoex::TFLRelu6>
{
public:
  void handover(loco::ReLU6 *origin, locoex::TFLRelu6 *replacer) override
  {
    replacer->features(origin->input());
  }

  std::vector<loco::Node *> getInputsToConvert(loco::ReLU6 *origin) override
  {
    std::vector<loco::Node *> inputs({origin->input()});
    return inputs;
  }

  void set(locoex::TFLRelu6 *replacer, std::vector<loco::Node *> &to) override
  {
    assert(to.size() == 1);

    replacer->features(to.at(0));
  }

  void nullify(loco::ReLU6 *origin) override { origin->input(nullptr); }
};

} // namespace

namespace exo
{

bool Relu6Converter::convert(loco::ReLU6 *origin)
{
  Relu6InputHandler input_handler;
  exo::DomainConverter<loco::ReLU6, locoex::TFLRelu6> domain_converter;

  auto tfl_node = domain_converter.convert<FeatureLayout::NHWC>(origin, input_handler);

  return (tfl_node != nullptr);
}

} // namespace exo
