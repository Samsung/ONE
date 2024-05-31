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

namespace onert
{
namespace compiler
{

/**
 * @brief Class to compile NN package
 */
class Compiler : public ICompiler
{
public:
  /**
   * @brief     Construct a new Compiler object for single model
   * @param[in] model model to compile
   * @param[in] copts Compiler Options
   */
  Compiler(const std::shared_ptr<ir::Model> &model, CompilerOptions *copts);

  /**
   * @brief     Construct a new Compiler object for NN package
   * @param[in] nnpkg NN package to compile
   * @param[in] copts Compiler option for package
   */
  Compiler(const std::shared_ptr<ir::NNPkg> &nnpkg, CompilerOptions *copts);

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
