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

#ifndef __CONVERSION_ELTWISEBINARY_CONVERTER_H__
#define __CONVERSION_ELTWISEBINARY_CONVERTER_H__

#include "GraphBlock.h"
#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco/IR/Nodes.h>

#include <loco/Service/ShapeInference.h>

namespace
{

template <class ELTWISEBIN, class TFLBIN>
class EltwiseBinInputHandler : public exo::InputHandler<ELTWISEBIN, TFLBIN>
{
public:
  void handover(ELTWISEBIN *origin, TFLBIN *replacer) override
  {
    assert(origin && replacer);
    replacer->x(origin->lhs());
    replacer->y(origin->rhs());
  }

  std::vector<loco::Node *> getInputsToConvert(ELTWISEBIN *origin) override
  {
    assert(origin);
    std::vector<loco::Node *> inputs({origin->lhs(), origin->rhs()});
    return inputs;
  }

  void set(TFLBIN *replacer, std::vector<loco::Node *> &to) override
  {
    assert(to.size() == 2);

    replacer->x(to.at(0));
    replacer->y(to.at(1));
  }

  void nullify(ELTWISEBIN *origin) override
  {
    assert(origin);
    origin->lhs(nullptr);
    origin->rhs(nullptr);
  }
};

template <class TFLBIN> void init_fused_act_func(TFLBIN *);

template <> inline void init_fused_act_func(locoex::TFLAdd *node)
{
  node->fusedActivationFunction(locoex::FusedActFunc::NONE);
}

template <> inline void init_fused_act_func(locoex::TFLMul *node)
{
  node->fusedActivationFunction(locoex::FusedActFunc::NONE);
}

template <> inline void init_fused_act_func(locoex::TFLSub *node)
{
  node->fusedActivationFunction(locoex::FusedActFunc::NONE);
}

template <> inline void init_fused_act_func(locoex::TFLDiv *node)
{
  node->fusedActivationFunction(locoex::FusedActFunc::NONE);
}

} // namespace

namespace exo
{

template <class ELTWISEBIN, class TFLBIN> bool EltwiseBinaryConvert(ELTWISEBIN *origin)
{
  EltwiseBinInputHandler<ELTWISEBIN, TFLBIN> input_handler;
  exo::DomainConverter<ELTWISEBIN, TFLBIN> domain_converter;

  auto tfl_node = domain_converter.template convert<FeatureLayout::NHWC>(origin, input_handler);

  if (tfl_node == nullptr)
    return false;

  init_fused_act_func(tfl_node);

  return true;
}

} // namespace exo

#endif // __CONVERSION_ELTWISEBINARY_CONVERTER_H__
