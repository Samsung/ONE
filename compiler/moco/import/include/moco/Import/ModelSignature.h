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

#ifndef __MOCO_IMPORT_MODELSIGNATURE_H__
#define __MOCO_IMPORT_MODELSIGNATURE_H__

#include <moco/Names.h>

#include <loco.h>
#include <angkor/TensorShape.h>

#include <string>
#include <vector>

namespace moco
{

/**
 * @brief Class to store information to run a model. Normally this info comes from users
 *        via CLI params or configuration file.
 */
struct ModelSignature
{
public:
  void add_input(const TensorName &input) { _inputs.push_back(input); }
  void add_input(const TensorName &&input) { _inputs.push_back(input); }
  void add_output(const TensorName &output) { _outputs.push_back(output); }
  void add_output(const TensorName &&output) { _outputs.push_back(output); }

  const std::vector<TensorName> &inputs() const { return _inputs; }
  const std::vector<TensorName> &outputs() const { return _outputs; }

  /**
   * @brief Adds customop op type (not name of node) provided from user
   */
  void add_customop(const std::string &op);
  const std::vector<std::string> &customops() const { return _customops; }

  /**
   * @brief Adds node name and its shape provided from user
   */
  void shape(const std::string &node_name, const angkor::TensorShape &shape);
  const angkor::TensorShape *shape(const std::string &node_name) const;

  /**
   * @brief Adds node name and its dtype provided from user
   */
  void dtype(const std::string &node_name, loco::DataType dtype);
  loco::DataType dtype(const std::string &node_name) const;

private:
  std::vector<TensorName> _inputs;  // graph inputs
  std::vector<TensorName> _outputs; // graph outputs

  // For custom op types passed from user (e.g., via CLI)
  std::vector<std::string> _customops;

  // For and node names and shapes passed from user (e.g., via CLI)
  std::map<std::string, angkor::TensorShape> _shapes;

  // For and node names and dtype passed from user (e.g., via CLI)
  std::map<std::string, loco::DataType> _dtypes;
};

} // namespace moco

#endif // __MOCO_IMPORT_MODELSIGNATURE_H__
