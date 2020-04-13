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

#include "pass/PassException.h"

#include "gtest/gtest.h"

using namespace nnc;

namespace
{

std::string ErrorMsg = "error constructor";

void passErr1() { throw PassException(ErrorMsg); }

void passErr2()
{
  try
  {
    passErr1();
  }
  catch (const PassException &e)
  {
    throw;
  }
}

TEST(CONTRIB_PASS, PassException)
{
  try
  {
    passErr2();
  }
  catch (const PassException &e)
  {
    ASSERT_TRUE(ErrorMsg == e.what());
    return;
  }

  // should not happen
  FAIL();
}

} // unnamed namespace
