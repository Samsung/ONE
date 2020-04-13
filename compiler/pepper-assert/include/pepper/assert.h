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

#ifndef __PEPPER_ASSERT_H__
#define __PEPPER_ASSERT_H__

#include <cassert>

//
// This example shows how to use DBGARG macro.
//
// void f(DBGARG(uint32_t, n))
// {
//   assert(n < 128);
// }
//
// This will make it easy to remove unused variable warnings in Release build.
//
#ifdef NDEBUG
#define DBGARG(TYP, VAR) TYP
#else
#define DBGARG(TYP, VAR) TYP VAR
#endif // NDEBUG

#endif // __PEPPER_ASSERT_H__
