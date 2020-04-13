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

#ifndef __LOCOEX_COPNODE_DECL_H__
#define __LOCOEX_COPNODE_DECL_H__

#include <loco/IR/Node.h>
#include <loco/IR/Dialect.h>

namespace locoex
{

struct COpNode : public loco::Node
{
  virtual ~COpNode() = default;

  const loco::Dialect *dialect(void) const final;

  uint32_t opnum(void) const final { return 0; /* opnum for custom op */ }
};

} // namespace locoex

#endif // __LOCOEX_COPNODE_DECL_H__
