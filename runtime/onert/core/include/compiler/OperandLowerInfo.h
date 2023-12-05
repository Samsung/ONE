/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_COMPILER_OPERAND_LOWER_INFO_H__
#define __ONERT_COMPILER_OPERAND_LOWER_INFO_H__

#include <functional>
#include <stdint.h>

#include "compiler/PermuteFactor.h"
#include "ir/Layout.h"
#include "util/Set.h"

namespace onert
{
namespace backend
{
class Backend;
} // namespace backend
} // namespace onert

namespace onert
{
namespace compiler
{

using PermuteFactorSet = util::Set<PermuteFactor>;

class OperandLowerInfo
{
public:
  OperandLowerInfo()
  {
    // DO NOTHING
  }

public:
  const PermuteFactorSet &def_factors(void) const { return _def_factors; }
  const PermuteFactorSet &use_factors(void) const { return _use_factors; }
  ir::Layout layout(void) const { return _layout; }

public:
  void addDefPermuteFactor(const PermuteFactor &factor) { _def_factors.add(factor); }
  void addUsePermuteFactor(const PermuteFactor &factor) { _use_factors.add(factor); }
  void removeDefPermuteFactor(const PermuteFactor &factor) { _def_factors.remove(factor); }
  void removeUsePermuteFactor(const PermuteFactor &factor) { _use_factors.remove(factor); }
  void setLayout(ir::Layout layout) { _layout = layout; }

private:
  PermuteFactorSet _def_factors;
  PermuteFactorSet _use_factors;
  ir::Layout _layout;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_OPERAND_LOWER_INFO_H__
