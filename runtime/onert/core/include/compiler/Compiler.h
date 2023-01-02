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

/**
 * @file  Compiler.h
 * @brief This file contains Compiler class to define and run compilation phase
 */

#ifndef __ONERT_COMPILER_COMPILE_H_
#define __ONERT_COMPILER_COMPILE_H_

#include "CompilerOptions.h"
#include "ICompiler.h"
#include "ir/NNPkg.h"
#include "exec/Executors.h"
#include "util/TracingCtx.h"

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

private:
  std::shared_ptr<ir::Graph> &primary_subgraph()
  {
    return _nnpkg->primary_model()->at(ir::SubgraphIndex{0});
  }

private:
  std::shared_ptr<ir::NNPkg> _nnpkg;
  std::vector<CompilerOptions *> _voptions;
};

/**
 * @brief Class to compile NN package
 */
class Compiler : public ICompiler
{
public:
  /**
   * @brief     Construct a new Compiler object for single model
   * @param[in] model     model to compile
   * @param[in] coptions  Compiler Options
   */
  Compiler(const std::shared_ptr<ir::Model> &model, CompilerOptions &copt);

  /**
   * @brief     Construct a new Compiler object for NN package
   * @param[in] nnpkg    NN package to compile
   * @param[in] coptions Compiler option vector for each model in package
   */
  Compiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
           std::vector<std::unique_ptr<CompilerOptions>> &copts);

  /**
   * @brief Destroy the Compiler object
   */
  ~Compiler() = default;

public:
  /**
   * @brief   Do compilation with the options
   *
   * @return std::shared_ptr<CompilerArtifact> Executors as a result of compilation
   */
  std::shared_ptr<CompilerArtifact> compile(void);

private:
  std::shared_ptr<ir::Model> _model;
  CompilerOptions *_options;
};

} // namespace compiler

} // namespace onert

#endif // __ONERT_COMPILER_COMPILE_H_
