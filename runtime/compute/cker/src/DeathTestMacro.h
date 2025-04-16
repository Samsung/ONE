/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __DEATH_TEST_MACROS_H__
#define __DEATH_TEST_MACROS_H__

#include <gtest/gtest.h>

// In release mode, assertions might not trigger abort() via assert(),
// so we use EXPECT_EXIT and require that the statement exits with EXIT_FAILURE.
// In debug mode, we use EXPECT_DEATH as usual.
#ifdef NDEBUG
#define EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(statement, regex)
#else
#define EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(statement, regex) \
  EXPECT_EXIT(statement, ::testing::KilledBySignal(SIGABRT), regex)
#endif

#endif // __DEATH_TEST_MACROS_H__
