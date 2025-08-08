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

#ifndef __ONERT_COMPILER_COMPILER_FACTORY_H__
#define __ONERT_COMPILER_COMPILER_FACTORY_H__

#include "ICompiler.h"
#include "CompilerOptions.h"
#include "ir/NNPkg.h"
#include "ir/train/TrainingInfo.h"

namespace onert::compiler
{

// TODO Support register and use compiler plugin
class CompilerFactory
{
public:
  static CompilerFactory &get();

public:
  /**
   * @brief     Create ICompiler instance. Ownership of nnpkg is moved to ICompiler instance
   *
   * Compiler is designed on assumption that caller will not use nnpkg after calling this function.
   * So ownership of nnpkg is moved to ICompiler instance to handle memory overhead of nnpkg.
   * If caller want to maintain nnpkg after calling this function,
   * caller should clone nnpkg before calling this function.
   *
   * @param[in] nnpkg         Package to compile
   * @param[in] copts         Compiler options
   * @param[in] training_info Training info if it is a training model, otherwise nullptr
   * @return    ICompiler instance pointer which owns nnpkg
   */
  std::unique_ptr<ICompiler> create(std::unique_ptr<ir::NNPkg> nnpkg, CompilerOptions *copts,
                                    const ir::train::TrainingInfo *training_info = nullptr);

private:
  // It is not allowed to use CompilerFactory without get()
  CompilerFactory() = default;
};

} // namespace onert::compiler

#endif // __ONERT_COMPILER_COMPILER_FACTORY_H__
