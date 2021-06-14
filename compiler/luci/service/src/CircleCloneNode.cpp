/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CircleCloneNode.h"

namespace luci
{

luci::CircleNode *CloneNode::visit(const luci::CircleNode *node)
{
#define CNVISIT_GRP(GRP)              \
  {                                   \
    CloneNodeLet<CN::GRP> cn(_graph); \
    auto cloned = node->accept(&cn);  \
    if (cloned != nullptr)            \
      return cloned;                  \
  }

  CNVISIT_GRP(ABC);
  CNVISIT_GRP(DEF);
  CNVISIT_GRP(GHIJ);
  CNVISIT_GRP(KLMN);
  CNVISIT_GRP(OPQR);
  CNVISIT_GRP(STUV);
  CNVISIT_GRP(WXYZ);

  return nullptr;
}

} // namespace luci
