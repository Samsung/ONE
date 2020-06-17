/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_LANG_DEADNODEQUERYSERVICE_H__
#define __LUCI_LANG_DEADNODEQUERYSERVICE_H__

#include <logo/DeadNodeQueryService.h>

#include <loco/IR/Node.h>

namespace luci
{

struct DeadNodeQueryServiceImpl final : public logo::DeadNodeQueryService
{
  bool isDeadNode(loco::Node *node) final;
};

} // namespace luci

#endif // __LUCI_LANG_DEADNODEQUERYSERVICE_H__
