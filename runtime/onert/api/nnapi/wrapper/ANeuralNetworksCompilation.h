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

#ifndef __COMPILATION_H__
#define __COMPILATION_H__

#include "ANeuralNetworksModel.h"

#include "compiler/Compiler.h"
#include "ir/Graph.h"
#include "ir/Model.h"
#include "exec/IExecutors.h"
#include "util/TracingCtx.h"

struct ANeuralNetworksCompilation
{
public:
  ANeuralNetworksCompilation(const ANeuralNetworksModel *model);

public:
  bool finish() noexcept;
  bool isFinished() noexcept { return _compiler == nullptr; }

  void publish(std::shared_ptr<onert::exec::IExecutors> &executors) noexcept
  {
    executors = _artifact ? _artifact->_executors : nullptr;
  }

private:
  std::shared_ptr<onert::ir::Model> _model;
  std::unique_ptr<onert::compiler::CompilerOptions> _coptions;
  std::shared_ptr<onert::compiler::Compiler> _compiler;
  std::shared_ptr<onert::compiler::CompilerArtifact> _artifact;
};

#endif
