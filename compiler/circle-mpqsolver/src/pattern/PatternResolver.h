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

#ifndef __MPQSOLVER_PATTERN_RESOLVER_H__
#define __MPQSOLVER_PATTERN_RESOLVER_H__

#include <luci/CircleQuantizer.h>
#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>

#include <map>

namespace mpqsolver
{
namespace pattern
{

class PatternResolver
{
public:
  virtual ~PatternResolver() = default;
  virtual std::map<luci::CircleNode *, luci::CircleQuantizer::Options::LayerParam>
  resolve(const luci::Module *module) = 0;
};

class Q8LayerNormWithQ16VarianceResolver : public PatternResolver
{
public:
  /**
   * @brief resolve all nodes of LayerNorm pattern as prescribed
   */
  std::map<luci::CircleNode *, luci::CircleQuantizer::Options::LayerParam>
  resolve(const luci::Module *module) override;
};

class Q8SoftmaxWithQ16SubExpResolver : public PatternResolver
{
public:
  /**
   * @brief resolve all nodes of Softmax pattern as prescribed
   */
  std::map<luci::CircleNode *, luci::CircleQuantizer::Options::LayerParam>
  resolve(const luci::Module *module) override;
};

} // namespace pattern
} // namespace mpqsolver

#endif //__MPQSOLVER_PATTERN_RESOLVER_H__
