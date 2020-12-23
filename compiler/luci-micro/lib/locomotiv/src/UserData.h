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

#ifndef _LOCOMOTIV_USERDATA_H_
#define _LOCOMOTIV_USERDATA_H_

#include "locomotiv/NodeData.h"

namespace locomotiv
{

const NodeData *user_data(const loco::Node *node);
void user_data(loco::Node *node, std::unique_ptr<NodeData> &&data);
void erase_user_data(loco::Node *node);

} // namespace locomotiv

#endif // _LOCOMOTIV_USERDATA_H_
