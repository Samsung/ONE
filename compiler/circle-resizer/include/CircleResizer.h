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

#include <string>
#include <memory>
#include <vector>

namespace luci
{
class Module;
}

namespace circle_resizer
{
class CircleResizer
{
public:
  explicit CircleResizer(const std::string &model_path);
  // to satisfy forward declaration + unique_ptr
  ~CircleResizer();

public:
  void resize_model(const std::vector<Shape> &shapes);
  void save_model(const std::string &output_path) const;

public:
  std::vector<Shape> input_shapes() const;
  std::vector<Shape> output_shapes() const;

private:
  std::string _model_path;
  std::unique_ptr<luci::Module> _module;
};
} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_H__
