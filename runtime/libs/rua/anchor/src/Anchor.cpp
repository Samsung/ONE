/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "rua/Anchor.h"
#include <rua/DynamicBinder.h>

namespace
{

const rua::RuntimeService *anchored_service = rua::DynamicBinder::get();

} // namespace

namespace rua
{

const RuntimeService *Anchor::get(void) { return anchored_service; }
void Anchor::set(const RuntimeService *service) { anchored_service = service; }

} // namespace rua
