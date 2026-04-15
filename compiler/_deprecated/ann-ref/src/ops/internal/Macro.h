/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __COMPATIBILITY_H__
#define __COMPATIBILITY_H__

#include <cassert>
#include <cstdint>

#ifndef DCHECK
#define DCHECK(condition) assert((condition))
#endif

#ifndef DCHECK_EQ
#define DCHECK_EQ(x, y) assert((x) == (y))
#endif

#ifndef DCHECK_GE
#define DCHECK_GE(x, y) assert((x) >= (y))
#endif

#ifndef DCHECK_GT
#define DCHECK_GT(x, y) assert((x) > (y))
#endif

#ifndef DCHECK_LE
#define DCHECK_LE(x, y) assert((x) <= (y))
#endif

#ifndef DCHECK_LT
#define DCHECK_LT(x, y) assert((x) < (y))
#endif

#ifndef CHECK_EQ
#define CHECK_EQ(x, y) assert((x) == (y))
#endif

using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

#endif // __COMPATIBILITY_H__
