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

#ifndef __CIRCLE_OP_REGISTRY_H__
#define __CIRCLE_OP_REGISTRY_H__

#include "CircleOpChef.h"
#include "CircleOpChefs.h"

#include <memory>

namespace circlechef
{

/**
 * @brief circlechef operator registry
 */
class CircleOpRegistry
{
public:
  /**
   * @brief Returns registered CircleOpChef pointer for BuiltinOperator or
   *        nullptr if not registered
   */
  const CircleOpChef *lookup(circle::BuiltinOperator op) const
  {
    if (_circleop_map.find(op) == _circleop_map.end())
      return nullptr;

    return _circleop_map.at(op).get();
  }

  static CircleOpRegistry &get()
  {
    static CircleOpRegistry me;
    return me;
  }

private:
  CircleOpRegistry()
  {
#define REG_TFL_OP(OPCODE, CLASS) \
  _circleop_map[circle::BuiltinOperator_##OPCODE] = std::make_unique<CLASS>()

    REG_TFL_OP(BATCH_MATMUL, CircleOpBatchMatMul);
    REG_TFL_OP(BCQ_FULLY_CONNECTED, CircleOpBCQFullyConnected);
    REG_TFL_OP(BCQ_GATHER, CircleOpBCQGather);
    REG_TFL_OP(CIR_GRU, CircleOpCircleGRU);
    REG_TFL_OP(INSTANCE_NORM, CircleOpInstanceNorm);
#undef REG_TFL_OP
  }

private:
  std::map<circle::BuiltinOperator, std::unique_ptr<CircleOpChef>> _circleop_map;
};

} // namespace circlechef

#endif // __CIRCLE_OP_REGISTRY_H__
