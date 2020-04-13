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

#include "compiler/Compiler.h"
#include "ir/Graph.h"
#include "exec/IExecutor.h"

struct ANeuralNetworksCompilation
{
public:
  ANeuralNetworksCompilation(const std::shared_ptr<neurun::ir::Graph> &graph) noexcept;

public:
  bool finish() noexcept;

  neurun::compiler::State state(void) noexcept { return _compiler->state(); }
  void publish(std::shared_ptr<neurun::exec::IExecutor> &executor) noexcept
  {
    _compiler->release(executor);
  }

private:
  std::shared_ptr<neurun::compiler::Compiler> _compiler;
};

#endif
