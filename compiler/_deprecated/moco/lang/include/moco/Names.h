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

#ifndef __MOCO_NAMES_H__
#define __MOCO_NAMES_H__

#include <string>
#include <stdexcept>

namespace moco
{

struct TensorName final
{
public:
  /**
   * @brief Constructor
   *
   * @note If tensor_name does not have ":index", this constructor adds ":0" by default
   */
  explicit TensorName(const std::string &tensor_name)
  {
    if (tensor_name.find(":") != std::string::npos) // tensor_name is a form of letter:0
    {
      _name.assign(tensor_name);
    }
    else
    {
      _name.assign(tensor_name + ":0"); // if it does not have ":index", adds ":0" by default
    }
  }

  explicit TensorName(const std::string &node_name, const int tensor_index)
  {
    if (node_name.find(":") != std::string::npos) // tensor_name is already a form of name:0
    {
      // TODO including oops will make oops dependent to modules that include this
      // postpone decision to this or not
      throw std::runtime_error("Error: Node name has already tensor index:" + node_name);
    }
    else
    {
      _name.assign(node_name + ":" + std::to_string(tensor_index));
    }
  }

  const std::string &name() const { return _name; }

  /**
   * @brief Returns node name from tensor name by removing, e.g., ":0"
   */
  const std::string nodeName() const
  {
    auto index = _name.find(":");

    if (index != std::string::npos)
      return _name.substr(0, index);
    else
    {
      // TODO including oops will make oops dependent to modules that include this
      // postpone decision to this or not
      throw std::runtime_error{"Error: Tensor name should be a 'name:number' format: " + _name};
    }
  };

private:
  std::string _name;
};

/**
 * @brief To use TensorName as a key in std::map, this struct defines how to compare two TensorNames
 */
struct TensorNameCompare
{
  bool operator()(const TensorName &lhs, const TensorName &rhs) const
  {
    return lhs.name() < rhs.name();
  }
};

} // namespace moco

#endif // __MOCO_NAMES_H__
