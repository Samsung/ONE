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

#ifndef __CIRCLE_RESIZER_MODEL_H__
#define __CIRCLE_RESIZER_MODEL_H__

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
// DESIGN NOTE: The purpose of the class is to keep buffer and module synchronized
class ModelData
{
public:
  explicit ModelData(const std::vector<uint8_t> &buffer);
  explicit ModelData(const std::string &model_path);
  // to satisfy forward declaration + unique_ptr
  ~ModelData();

  void invalidate_module();
  void invalidate_buffer();

  std::vector<uint8_t> &buffer();
  luci::Module *module();

  std::vector<Shape> input_shapes();
  std::vector<Shape> output_shapes();

  void save(std::ostream &stream);
  void save(const std::string &output_path);

private:
  bool _module_invalidated = false, _buffer_invalidated = false;
  std::vector<uint8_t> _buffer;
  std::unique_ptr<luci::Module> _module;
};
} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_H__
