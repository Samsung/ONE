/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <gtest/gtest.h>

#include "DumpingHooks.h"
#include "core/TestHelper.h"

#include <ftw.h>
#include <string>

namespace
{

class CircleMPQSolverDumpingHooksTest : public ::testing::Test
{
public:
  CircleMPQSolverDumpingHooksTest()
  {
    char folderTemplate[] = "CircleMPQSolverDumpingHooksTestXXXXXX";
    _folder = mpqsolver::test::io_utils::makeTemporaryFolder(folderTemplate);
  }

  ~CircleMPQSolverDumpingHooksTest()
  {
    // cleanup
    auto callback = [](const char *child, const struct stat *, int, struct FTW *) {
      return remove(child);
    };
    nftw(_folder.c_str(), callback, 128, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
  }

protected:
  std::string _folder;
};

} // namespace

TEST_F(CircleMPQSolverDumpingHooksTest, verifyResultsTest)
{
  mpqsolver::core::DumpingHooks hooks(_folder);
  EXPECT_NO_THROW(hooks.on_begin_solver("model_path.circle", 0.0, 1.0));
  std::string errors_path = _folder + "/errors" + ".mpq.txt";
  EXPECT_TRUE(mpqsolver::test::io_utils::isFileExists(errors_path));

  hooks.on_begin_iteration();

  EXPECT_NO_THROW(hooks.on_end_iteration(mpqsolver::core::LayerParams(), "uint8", 0.0));
  std::string current_mpq_path = _folder + "/Configuration_" + std::to_string(1) + ".mpq.json";
  EXPECT_TRUE(mpqsolver::test::io_utils::isFileExists(current_mpq_path));

  EXPECT_NO_THROW(hooks.on_end_solver(mpqsolver::core::LayerParams(), "uint8", 0.5));
  std::string final_mpq_path = _folder + "/FinalConfiguration" + ".mpq.json";
  EXPECT_TRUE(mpqsolver::test::io_utils::isFileExists(final_mpq_path));
}

TEST_F(CircleMPQSolverDumpingHooksTest, empty_path_NEG)
{
  mpqsolver::core::DumpingHooks hooks("");
  EXPECT_ANY_THROW(hooks.on_begin_solver("", -1, -1));
  hooks.on_begin_iteration();
  EXPECT_ANY_THROW(hooks.on_quantized(nullptr));
  EXPECT_ANY_THROW(hooks.on_end_iteration(mpqsolver::core::LayerParams(), "uint8", -1));
  EXPECT_ANY_THROW(hooks.on_end_solver(mpqsolver::core::LayerParams(), "uint8", -1));
}
