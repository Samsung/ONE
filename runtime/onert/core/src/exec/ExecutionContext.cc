/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exec/ExecutionContext.h"

#include "util/ConfigSource.h"

namespace onert
{
namespace exec
{

std::unique_ptr<ExecutionOptions> ExecutionOptions::fromGlobalConfig()
{
  auto options = std::make_unique<ExecutionOptions>();
  options->dump_minmax = util::getConfigBool(util::config::MINMAX_DUMP);
  options->trace = util::getConfigBool(util::config::TRACING_MODE);
  options->profile = util::getConfigBool(util::config::PROFILING_MODE);

  return options;
}

} // namespace exec
} // namespace onert
