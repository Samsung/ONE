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

namespace onert
{
namespace compiler
{

// TODO Support register and use compiler plugin
class CompilerFactory
{
public:
  static CompilerFactory &get();

public:
  std::unique_ptr<ICompiler> create(const std::shared_ptr<ir::NNPkg> &nnpkg, CompilerOptions *copts,
                                    const ir::train::TrainingInfo *training_info = nullptr);

private:
  // It is not allowed to use CompilerFactory without get()
  CompilerFactory() = default;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_COMPILER_FACTORY_H__
