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

#include "PassRunner.h"

namespace onert
{
namespace compiler
{
namespace pass
{

PassRunner &PassRunner::append(std::unique_ptr<IPass> pass)
{
  _passes.emplace_back(std::move(pass));
  return *this;
}

void PassRunner::run()
{
  for (auto &&pass : _passes)
  {
    VERBOSE(PassRunner) << "Start running '" << pass->id() << "'" << std::endl;
    pass->run();
    VERBOSE(PassRunner) << "Finished running '" << pass->id() << "'" << std::endl;
    // TODO Dump graph?
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
