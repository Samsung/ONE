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

#ifndef __CIRCLE_RESIZER_MODEL_DATA_H__
#define __CIRCLE_RESIZER_MODEL_DATA_H__

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

/**
 * The representation of Circle Model.
 * The purpose of the class is to keep the buffer and the module representation of the model
 * synchronized.
 */
class ModelData
{
public:
  /**
   * @brief Initialize the model with buffer representation.
   *
   * Exceptions:
   * - std::runtime_error if interpretation of provided buffer as a circle model failed.
   */
  explicit ModelData(const std::vector<uint8_t> &buffer);

  /**
   * @brief Initialize the model with buffer representation.
   *
   * Exceptions:
   * - std::runtime_error if reading a model from provided path failed.
   */
  explicit ModelData(const std::string &model_path);

  /**
   * @brief Dtor of ModelData. Note that explicit declaration is needed to satisfy forward
   * declaration + unique_ptr.
   */
  ~ModelData();

  /**
   * @brief Notify that the buffer representation of the model has been modified so the module is no
   * more valid.
   */
  void invalidate_module();

  /**
   * @brief Notify that the module representation of the model has been modified so the buffer is no
   * more valid.
   */
  void invalidate_buffer();

  /**
   * @brief Get the loaded model as the buffer.
   */
  std::vector<uint8_t> &buffer();

  /**
   * @brief Get the loaded model as the module.
   */
  luci::Module *module();

  /**
   * @brief Get input shapes of the loaded model.
   */
  std::vector<Shape> input_shapes();

  /**
   * @brief Get output shapes of the loaded model.
   *
   */
  std::vector<Shape> output_shapes();

  /**
   * @brief Save the loaded model to the stream.
   *
   * Exceptions:
   * - std::runtime_error if saving the model the given stream failed.
   */
  void save(std::ostream &stream);

  /**
   * @brief Save the loaded model to the location indicated by output_path.
   *
   * Exceptions:
   * - std::runtime_error if saving the model the given path failed.
   */
  void save(const std::string &output_path);

private:
  bool _module_invalidated = false, _buffer_invalidated = false;
  std::vector<uint8_t> _buffer;
  std::unique_ptr<luci::Module> _module;
};

} // namespace circle_resizer

#endif // __CIRCLE_RESIZER_MODEL_DATA_H__
