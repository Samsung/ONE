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

#include "compiler/CompilerFactory.h"

#include "Compiler.h"
#include "MultiModelCompiler.h"
#include "train/TrainingCompiler.h"

namespace onert::compiler
{

CompilerFactory &CompilerFactory::get()
{
  static CompilerFactory singleton;
  return singleton;
}

std::unique_ptr<ICompiler> CompilerFactory::create(std::unique_ptr<ir::NNPkg> nnpkg,
                                                   CompilerOptions *copts,
                                                   const ir::train::TrainingInfo *training_info)
{
  // Returing compiler for training
  if (training_info)
    return std::make_unique<train::TrainingCompiler>(std::move(nnpkg), copts, *training_info);

  // Returing compiler for inference
  return std::make_unique<MultiModelCompiler>(std::move(nnpkg), copts);
}

} // namespace onert::compiler
