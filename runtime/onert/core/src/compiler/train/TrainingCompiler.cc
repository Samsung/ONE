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

#include "TrainingCompiler.h"

#include "util/Utils.h"

namespace onert
{
namespace compiler
{
namespace train
{

TrainingCompiler::TrainingCompiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                                   std::vector<std::unique_ptr<CompilerOptions>> &copts,
                                   const TrainingInfo *training_info)
  : _model{nnpkg->primary_model()}, _options{copts[0].get()}, _training_info{training_info}
{
  if (nnpkg->model_count() > 1)
    throw std::runtime_error("TrainingCompiler does not support multiple models yet");

  if (nnpkg->primary_model()->subgraphs_count() > 1)
    throw std::runtime_error("TrainingCompiler does not support multiple subgraphs yet");
}

std::shared_ptr<CompilerArtifact> TrainingCompiler::compile(void)
{
  // TODO Implement

  // Avoid unused-private-field error
  UNUSED_RELEASE(_model);
  UNUSED_RELEASE(_inference_compiler);
  UNUSED_RELEASE(_options);
  UNUSED_RELEASE(_training_info);

  return std::make_shared<CompilerArtifact>(nullptr, nullptr);
}

} // namespace train
} // namespace compiler
} // namespace onert
