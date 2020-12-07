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

#include "ANeuralNetworksCompilation.h"

#include "util/logging.h"

// TODO Support multiple subgraphs
ANeuralNetworksCompilation::ANeuralNetworksCompilation(const ANeuralNetworksModel *model) noexcept
  : _subgraphs{model->getSubGraphs()}, _tracing_ctx{std::make_unique<onert::util::TracingCtx>(
                                         _subgraphs.get())},
    _compiler{new onert::compiler::Compiler{_subgraphs, _tracing_ctx.get()}}
{
  if (model->allowedToFp16())
  {
    _compiler->enableToFp16();
  }
}

bool ANeuralNetworksCompilation::finish() noexcept
{
  try
  {
    _executors = _compiler->compile();
  }
  catch (const std::exception &e)
  {
    VERBOSE(EXCEPTION) << e.what() << std::endl;

    return false;
  }

  return true;
}
