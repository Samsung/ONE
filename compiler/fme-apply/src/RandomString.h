/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __FME_APPLY_RANDOM_STRING_H__
#define __FME_APPLY_RANDOM_STRING_H__

#include <cstdint>
#include <string>

namespace fme_apply
{

// Generate random string composed of alphabet + numeric
// Use length 6 by default
// NOTE This is used to generate unique ID
std::string random_str(uint32_t len = 6);

} // namespace fme_apply

#endif //__FME_APPLY_RANDOM_STRING_H__
