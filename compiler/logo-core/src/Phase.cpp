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

#include <logo/Phase.h>

namespace logo
{

void PhaseRunner<PhaseStrategy::Saturate>::run(const Phase &phase) const
{
  notifyPhaseBegin();

  for (bool changed = true; changed;)
  {
    changed = false;

    for (auto &pass : phase)
    {
      notifyPassBegin(pass.get());

      bool pass_changed = pass->run(_graph);
      changed = changed || pass_changed;

      notifyPassEnd(pass.get(), pass_changed);
    }
  }

  notifyPhaseEnd();
}

void PhaseRunner<PhaseStrategy::Restart>::run(const Phase &phase) const
{
  notifyPhaseBegin();

  for (bool changed = true; changed;)
  {
    changed = false;

    for (auto &pass : phase)
    {
      notifyPassBegin(pass.get());

      bool pass_changed = pass->run(_graph);
      changed = changed || pass_changed;

      notifyPassEnd(pass.get(), pass_changed);

      if (changed)
      {
        break;
      }
    }
  }

  notifyPhaseEnd();
}

} // namespace logo
