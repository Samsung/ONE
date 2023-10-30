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

#ifndef __MPQSOLVER_PATTERN_SOLVER_H__
#define __MPQSOLVER_PATTERN_SOLVER_H__

#include "MPQSolver.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mpqsolver
{
namespace pattern
{

enum class QuantizationPattern
{
  Q8LayerNormWithQ16Variance,
  Q8SoftmaxWithQ16SubExp,
};

struct MPQOptions
{
  std::vector<QuantizationPattern> _patterns;
};

struct FrozenNodes
{
  std::map<luci::CircleNode *, luci::CircleQuantizer::Options::LayerParam> _node_to_param;
};

class PatternSolver final : public MPQSolver
{
public:
  /**
   * @brief construct PatternSolver using input_quantization/output_quantization to set
   * quantization type at input/output respectively and patterns to apply
   */
  PatternSolver(const std::string &input_quantization, const std::string &output_quantization,
                const std::vector<QuantizationPattern> &patterns);

  /**
   * @brief run solver for recorded float module at module_path
   */
  std::unique_ptr<luci::Module> run(const std::string &module_path) override;

private:
  /**
   * @brief set quantization options
   */
  void set_mpq_options(MPQOptions &options);

  /**
   * @brief fill _frozen with prescribed quantization parameters of resolved nodes
   */
  void resolve_patterns(luci::Module *module);

  /**
   * @brief transform _frozen nodes to Quantizer friendly form
   */
  luci::CircleQuantizer::Options::LayerParams get_frozen_params() const;

private:
  MPQOptions _options; // options for mpq quantization
  FrozenNodes _frozen; // nodes with prescribed quantization parameters
};

} // namespace pattern
} // namespace mpqsolver

#endif //__MPQSOLVER_PATTERN_SOLVER_H__
