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

#ifndef __TFLITE_OP_CHEF_H__
#define __TFLITE_OP_CHEF_H__

#include <mio/tflite/schema_generated.h>

#include <tflchef.pb.h>

#include "TFliteImport.h"

namespace tflchef
{

struct RecipeChefContext
{
  const tflite::Operator *tflop = nullptr;
  tflchef::Operation *chefop = nullptr;
  // add more if needed
};

/**
 * @brief Interface for each operators to build tflchef
 */
class TFliteOpChef
{
public:
  virtual void filler(const tflite::Operator *op, TFliteImport *import,
                      tflchef::ModelRecipe *model_recipe) const = 0;
  virtual ::tflchef::Operation *build(RecipeChefContext *ctx) const = 0;
  virtual ~TFliteOpChef() {}
};

} // namespace tflchef

#endif // __TFLITE_OP_CHEF_H__
