/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_PASS_PASS_RUNNER_H__
#define __ONERT_COMPILER_PASS_PASS_RUNNER_H__

#include <initializer_list>
#include <memory>
#include <vector>

#include "Pass.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

/**
 * @brief Composite passes with logging
 */
class PassRunner
{
public:
  PassRunner() = default;
  PassRunner &append(std::unique_ptr<IPass> pass);

  void run();

private:
  std::vector<std::unique_ptr<IPass>> _passes;
};

} // namespace pass
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_PASS_PASS_RUNNER_H__
