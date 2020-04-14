/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFLITE_OP_REGISTRY_H__
#define __TFLITE_OP_REGISTRY_H__

#include "TFliteOpChef.h"
#include "TFliteOpChefs.h"

#include <stdex/Memory.h>

using stdex::make_unique;

namespace tflchef
{

/**
 * @brief tflchef operator registry
 */
class TFliteOpRegistry
{
public:
  /**
   * @brief Returns registered TFliteOpChef pointer for BuiltinOperator or
   *        nullptr if not registered
   */
  const TFliteOpChef *lookup(tflite::BuiltinOperator op) const
  {
    if (_tfliteop_map.find(op) == _tfliteop_map.end())
      return nullptr;

    return _tfliteop_map.at(op).get();
  }

  static TFliteOpRegistry &get()
  {
    static TFliteOpRegistry me;
    return me;
  }

private:
  TFliteOpRegistry()
  {
    _tfliteop_map[tflite::BuiltinOperator_ABS] = make_unique<TFliteOpAbs>();
    _tfliteop_map[tflite::BuiltinOperator_ADD] = make_unique<TFliteOpAdd>();
    _tfliteop_map[tflite::BuiltinOperator_ARG_MAX] = make_unique<TFliteOpArgMax>();
    _tfliteop_map[tflite::BuiltinOperator_AVERAGE_POOL_2D] = make_unique<TFliteOpAveragePool2D>();
    _tfliteop_map[tflite::BuiltinOperator_CONCATENATION] = make_unique<TFliteOpConcatenation>();
    _tfliteop_map[tflite::BuiltinOperator_CONV_2D] = make_unique<TFliteOpConv2D>();
    _tfliteop_map[tflite::BuiltinOperator_COS] = make_unique<TFliteOpCos>();
    _tfliteop_map[tflite::BuiltinOperator_DEPTHWISE_CONV_2D] =
        make_unique<TFliteOpDepthwiseConv2D>();
    _tfliteop_map[tflite::BuiltinOperator_DIV] = make_unique<TFliteOpDiv>();
    _tfliteop_map[tflite::BuiltinOperator_EQUAL] = make_unique<TFliteOpEqual>();
    _tfliteop_map[tflite::BuiltinOperator_FLOOR_DIV] = make_unique<TFliteOpFloorDiv>();
    _tfliteop_map[tflite::BuiltinOperator_FULLY_CONNECTED] = make_unique<TFliteOpFullyConnected>();
    _tfliteop_map[tflite::BuiltinOperator_LOGICAL_NOT] = make_unique<TFliteOpLogicalNot>();
    _tfliteop_map[tflite::BuiltinOperator_LOGICAL_OR] = make_unique<TFliteOpLogicalOr>();
    _tfliteop_map[tflite::BuiltinOperator_MAX_POOL_2D] = make_unique<TFliteOpMaxPool2D>();
    _tfliteop_map[tflite::BuiltinOperator_MEAN] = make_unique<TFliteOpMean>();
    _tfliteop_map[tflite::BuiltinOperator_PACK] = make_unique<TFliteOpPack>();
    _tfliteop_map[tflite::BuiltinOperator_PAD] = make_unique<TFliteOpPad>();
    _tfliteop_map[tflite::BuiltinOperator_RELU] = make_unique<TFliteOpReLU>();
    _tfliteop_map[tflite::BuiltinOperator_RELU6] = make_unique<TFliteOpReLU6>();
    _tfliteop_map[tflite::BuiltinOperator_RESHAPE] = make_unique<TFliteOpReshape>();
    _tfliteop_map[tflite::BuiltinOperator_RSQRT] = make_unique<TFliteOpRsqrt>();
    _tfliteop_map[tflite::BuiltinOperator_SOFTMAX] = make_unique<TFliteOpSoftmax>();
    _tfliteop_map[tflite::BuiltinOperator_SQRT] = make_unique<TFliteOpSqrt>();
    _tfliteop_map[tflite::BuiltinOperator_SUB] = make_unique<TFliteOpSub>();
    _tfliteop_map[tflite::BuiltinOperator_TRANSPOSE] = make_unique<TFliteOpTranspose>();
  }

private:
  std::map<tflite::BuiltinOperator, std::unique_ptr<TFliteOpChef>> _tfliteop_map;
};

} // namespace tflchef

#endif // __TFLITE_OP_REGISTRY_H__
