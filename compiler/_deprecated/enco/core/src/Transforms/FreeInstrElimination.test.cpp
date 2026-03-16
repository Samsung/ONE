/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FreeInstrElimination.h"

#include <gtest/gtest.h>

TEST(FreeInstrEliminationTest, case_000)
{
  auto m = coco::Module::create();

  // Create a "free" Eval instruction
  m->entity()->instr()->create<coco::Eval>();

  ASSERT_EQ(m->entity()->instr()->size(), 1);

  // Apply "Free Instruction Elimination"
  enco::eliminate_free_instr(m.get());

  ASSERT_EQ(m->entity()->instr()->size(), 0);
}
