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

#include "oops/InternalExn.h"
#include "oops/UserExn.h"

#include <gtest/gtest.h>

namespace
{

void batman() { INTERNAL_EXN("Here comes Joker"); }

void star_wars() { INTERNAL_EXN_V("Something is approaching", "Darth Vader"); }

enum class InfinityStones
{
  SpaceStone,
  RealityStone,
  OtherStones,
};

void avengers()
{
  std::string where;
  std::string separator = ":";
  try
  {
    // exception will be raised in next line
    where = __FILE__ + separator + std::to_string(__LINE__ + 1);
    INTERNAL_EXN_V("Last stone was gathered", oops::to_uint32(InfinityStones::SpaceStone));
  }
  catch (const oops::InternalExn &e)
  {
    auto msg = std::string(e.what());
    ASSERT_TRUE(msg.find("Last stone was gathered: 0") != std::string::npos);
    ASSERT_TRUE(msg.find(where) != std::string::npos);
  }
}

} // namespace

TEST(oopsTest, InternalExn)
{
  ASSERT_THROW(batman(), oops::InternalExn);
  ASSERT_THROW(star_wars(), oops::InternalExn);

  avengers();
}

TEST(oopsTest, UserExn_one_info_after_msg)
{
  try
  {
    throw oops::UserExn("Not a member of Avenger", "Kingsman");
  }
  catch (const oops::UserExn &e)
  {
    auto msg = std::string(e.what());
    ASSERT_TRUE(msg.find("Not a member of Avenger: Kingsman") != std::string::npos);
  }
}

TEST(oopsTest, UserExn_two_pairs_after_msg)
{
  try
  {
    std::string hero("Spiderman");

    // clang-format off
    throw oops::UserExn("Hero's age is wrong",
                        "Hero", hero,
                        "Age", 97);
    // clang-format on
  }
  catch (const oops::UserExn &e)
  {
    auto msg = std::string(e.what());
    ASSERT_TRUE(msg.find("Hero = Spiderman, Age = 97") != std::string::npos);
  }
}
