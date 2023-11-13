/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file  TrainingCompiler.h
 * @brief This file contains TrainingCompiler class to define and run compilation phase
 */

#ifndef __ONERT_COMPILER_TRAIN_TRAINING_COMPILER_H_
#define __ONERT_COMPILER_TRAIN_TRAINING_COMPILER_H_

#include "compiler/CompilerOptions.h"
#include "compiler/ICompiler.h"
#include "compiler/train/TrainingInfo.h"
#include "ir/NNPkg.h"

namespace onert
{
namespace compiler
{
namespace train
{

/**
 * @brief Class to compile NN package
 */
class TrainingCompiler : public ICompiler
{
public:
  /**
   * @brief     Construct a new TrainingCompiler object for an nnpkg
   * @param[in] nnpkg         nnpkg to compile
   * @param[in] copts         compiler options
   * @param[in] training_info training information
   */
  explicit TrainingCompiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                            std::vector<std::unique_ptr<CompilerOptions>> &copts,
                            const TrainingInfo &training_info);

  /**
   * @brief Construct a TrainingCompiler object
   *
   */
  TrainingCompiler(void) = delete;

  /**
   * @brief Destroy the TrainingCompiler object
   */
  ~TrainingCompiler() = default;

public:
  /**
   * @brief Do compilation with the options
   *
   * @return std::shared_ptr<CompilerArtifact> Executors as a result of compilation
   */
  std::shared_ptr<CompilerArtifact> compile(void);

private:
  std::shared_ptr<ir::Model> _model;
  CompilerOptions *_options;
  const TrainingInfo _training_info;
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_TRAINING_COMPILER_H_
