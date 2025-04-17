/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_RESIZER_H__
#define __CIRCLE_RESIZER_H__

#include "Shape.h"
#include "CircleModel.h"

#include <string>
#include <vector>
#include <memory>

namespace circle_resizer
{
class ModelEditor
{
public:
  explicit ModelEditor(std::shared_ptr<CircleModel> circle_model);

public:
  ModelEditor &resize_inputs(const std::vector<Shape> &new_inputs_shapes);

private:
  std::shared_ptr<CircleModel> _circle_model;
};
} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_H__
