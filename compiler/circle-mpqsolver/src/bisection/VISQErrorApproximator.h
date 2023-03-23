/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MPQSOLVER_BISECTION_VISQ_ERROR_APPROXIMATOR_H__
#define __MPQSOLVER_BISECTION_VISQ_ERROR_APPROXIMATOR_H__

#include <string>
#include <map>

namespace mpqsolver
{
namespace bisection
{

class VISQErrorApproximator final
{
public:
  /**
   * @brief constructor of VISQErrorApproximator
   */
  VISQErrorApproximator() = default;

  /**
   * @brief initiliaze by visq_data_path (throws on failure)
   */
  void init(const std::string &visq_data_path);

  /**
   * @brief approximate error introduced while quantizing node into Q8
   */
  float approximate(const std::string &node_name) const;

private:
  /**
   * @brief initiliaze by visq_data (returns success)
   */
  bool init(std::istream &visq_data);

private:
  std::string _visq_data_path;
  std::map<std::string, float> _layer_errors;
};

} // namespace bisection
} // namespace mpqsolver

#endif // __MPQSOLVER_BISECTION_VISQ_ERROR_APPROXIMATOR_H__
