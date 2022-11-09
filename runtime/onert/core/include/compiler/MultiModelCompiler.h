/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_MULTI_MODEL_COMPILER_H_
#define __ONERT_COMPILER_MULTI_MODEL_COMPILER_H_

#include "ICompiler.h"
#include "ir/NNPkg.h"

namespace onert
{
namespace compiler
{

/**
 * @brief Class to compile NN package
 */
class MultiModelCompiler final : public ICompiler
{
public:
  /**
   * @brief     Construct a new Compiler object for NN package
   * @param[in] nnpkg    NN package to compile
   * @param[in] coptions Compiler option vector for each model in package
   */
  MultiModelCompiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                     std::vector<std::unique_ptr<CompilerOptions>> &copts);

  /**
   * @brief Destroy the MultiModelCompiler object
   */
  ~MultiModelCompiler() = default;

public:
  /**
   * @brief   Do compilation with the options
   *
   * @return std::shared_ptr<CompilerArtifact> Executors as a result of compilation
   */
  std::shared_ptr<CompilerArtifact> compile(void);

  /**
   * @brief   Allow to compute float32 using float16 data type
   */
  void enableToFp16();

private:
  void checkProfilerConditions();
  std::shared_ptr<ir::Graph> &primary_subgraph()
  {
    return _nnpkg->primary_model()->at(ir::SubgraphIndex{0});
  }

private:
  std::shared_ptr<ir::NNPkg> _nnpkg;
  std::vector<CompilerOptions *> _voptions;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_MULTI_MODEL_COMPILER_H_
