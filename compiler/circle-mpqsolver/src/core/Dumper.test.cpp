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

#include "Dumper.h"
#include "core/TestHelper.h"

#include <fstream>
#include <ftw.h>
#include <string>

TEST(CircleMPQSolverDumperTest, verifyResultsTest)
{
  char folderTemplate[] = "CircleMPQSolverDumperTestXXXXXX";
  auto const folder = mpqsolver::test::io_utils::makeTemporaryFolder(folderTemplate);
  mpqsolver::core::Dumper dumper(folder);
  dumper.set_model_path("");
  mpqsolver::core::LayerParams params;
  auto const step = 0;
  dumper.dump_MPQ_configuration(params, "uint8", step);

  std::string step_path = folder + "/Configuration_" + std::to_string(step) + ".mpq.json";
  EXPECT_TRUE(mpqsolver::test::io_utils::isFileExists(step_path));

  dumper.dump_final_MPQ(params, "uint8");
  std::string fin_path = folder + "/FinalConfiguration" + ".mpq.json";
  EXPECT_TRUE(mpqsolver::test::io_utils::isFileExists(fin_path));

  dumper.prepare_for_error_dumping();
  std::string errors_path = folder + "/errors" + ".mpq.txt";
  EXPECT_TRUE(mpqsolver::test::io_utils::isFileExists(errors_path));

  // cleanup
  auto callback = [](const char *child, const struct stat *, int, struct FTW *) {
    return remove(child);
  };
  nftw(folder.c_str(), callback, 128, FTW_DEPTH | FTW_MOUNT | FTW_PHYS);
}

TEST(CircleMPQSolverDumperTest, verifyResultsTest_NEG)
{
  mpqsolver::core::Dumper dumper("");
  dumper.set_model_path("");

  mpqsolver::core::LayerParams params;
  auto const step = 0;
  EXPECT_THROW(dumper.dump_MPQ_configuration(params, "uint8", step), std::runtime_error);
  EXPECT_THROW(dumper.dump_final_MPQ(params, "uint8"), std::runtime_error);
  EXPECT_THROW(dumper.prepare_for_error_dumping(), std::runtime_error);
}
