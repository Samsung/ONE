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

#ifndef __TFLITE_OP_UNIDIRECTIONALSEQUENCELSTM_H__
#define __TFLITE_OP_UNIDIRECTIONALSEQUENCELSTM_H__

#include "TFliteOpChef.h"

namespace tflchef
{

/**
 * @brief tflchef operator builder for UnidirectionalSequenceLSTM
 */
class TFliteOpUnidirectionalSequenceLSTM : public TFliteOpChef
{
public:
  void filler(const tflite::Operator *op, TFliteImport *import,
              tflchef::ModelRecipe *model_recipe) const override;
  tflchef::Operation *build(const tflite::Operator *op, TFliteImport *import,
                            tflchef::ModelRecipe *model_recipe) const override;
};

} // namespace tflchef

#endif // __TFLITE_OP_UNIDIRECTIONALSEQUENCELSTM_H__
