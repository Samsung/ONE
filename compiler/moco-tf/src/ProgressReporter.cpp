/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ProgressReporter.h"

#include "LogHelper.h"

#include <logo/Phase.h>
#include <logo/Pass.h>

#include <moco/Log.h>

#include <cassert>

namespace
{

char to_char(bool b) { return b ? 'Y' : 'N'; }

const char *to_str(logo::PhaseStrategy s)
{
  switch (s)
  {
    case logo::PhaseStrategy::Saturate:
      return "Saturate";
    case logo::PhaseStrategy::Restart:
      return "Restart";
  }
  assert(false);
  return "";
}

} // namespace

namespace moco
{
namespace tf
{

void ProgressReporter::notify(const logo::PhaseEventInfo<logo::PhaseEvent::PhaseBegin> *info)
{
  LOGGER(prime);

  INFO(prime) << "==============================================================";
  INFO(prime) << "PhaseRunner<" << to_str(strategy()) << ">";
  INFO(prime) << "Initial graph";
  INFO(prime) << moco::tf::fmt(graph());
}

void ProgressReporter::notify(const logo::PhaseEventInfo<logo::PhaseEvent::PhaseEnd> *info)
{
  LOGGER(prime);

  INFO(prime) << "PhaseRunner<" << to_str(strategy()) << "> - done";
}

void ProgressReporter::notify(const logo::PhaseEventInfo<logo::PhaseEvent::PassBegin> *info)
{
  LOGGER(prime);

  INFO(prime) << "--------------------------------------------------------------";
  INFO(prime) << "Before " << logo::pass_name(info->pass());
}

void ProgressReporter::notify(const logo::PhaseEventInfo<logo::PhaseEvent::PassEnd> *info)
{
  LOGGER(prime);

  INFO(prime) << "After " << logo::pass_name(info->pass())
              << " (changed: " << to_char(info->changed()) << ")";
  INFO(prime) << moco::tf::fmt(graph());
}

} // namespace tf
} // namespace moco
