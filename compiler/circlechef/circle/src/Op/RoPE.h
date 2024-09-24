/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_OP_ROPE_H__
#define __CIRCLE_OP_ROPE_H__

#include "CircleOpChef.h"

namespace circlechef
{

/**
 * @brief circlechef operator builder for RoPE
 */
class CircleOpRoPE : public CircleOpChef
{
public:
  void filler(const circle::Operator *op, CircleImport *import,
              circlechef::ModelRecipe *model_recipe) const override;
  circlechef::Operation *build(const circle::Operator *op, CircleImport *import,
                               circlechef::ModelRecipe *model_recipe) const override;
};

} // namespace circlechef

#endif // __CIRCLE_OP_ROPE_H__
