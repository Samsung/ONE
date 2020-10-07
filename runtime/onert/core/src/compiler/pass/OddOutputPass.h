/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_PASS_ODD_OUTPUT_PASS_H__
#define __ONERT_COMPILER_PASS_ODD_OUTPUT_PASS_H__

#include <unordered_set>

#include "Pass.h"
#include "ir/Index.h"

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief Pass to specially handle odd outputs in a subgraph
 *
 * Runtime Graph IR requires every input or output must have distinct tensor index, this is onert's
 * restriction. However we allow duplication of indices in the models(or API). So we should
 * transform the graph after model-loading.
 *
 * This is necessary since our API lets users to set different buffers for each input and output so
 * it is unavoidable that we must copy the value at runtime.
 *
 * Note that this is a mandatory pass for Graph.
 *
 * Case 1 : An operand which is a model output and a model input
 *
 * Create an operand and insert a Permute(copy) op between them. And change the output to be the
 * newly generated operand.
 *
 * e.g.)
 *
 * ```
 * ((#0 Input0 and also Output0))
 * becomes
 * ((#0 Input0)) -> [#0 Permute] -> ((#1 Output0))
 * ```
 *
 * Case 2 : Two or more duplicated outputs
 *
 * Do the same with Case 1, but between two outputs of the same tensor index.
 *
 * e.g.)
 *
 * ```
 * ((#0 Input0)) -> [#0 Some Operation] -> ((#1 Output0 and also Output1))
 * becomes
 * ((#0 Input0)) -> [#0 Some Operation] -> ((#1 Output0)) [#1 Permute] -> ((#2 Output1))
 * ```
 *
 */
class OddOutputPass : public Pass
{
public:
  using Pass::Pass;

public:
  std::string id() final { return "OddOutputPass"; }

public:
  void run() override;

private:
  ir::OperandIndex insertPermute(ir::OperandIndex input);
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_ODD_OUTPUT_PASS_H__
