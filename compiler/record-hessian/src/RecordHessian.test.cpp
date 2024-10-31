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

#include "record-hessian/RecordHessian.h"

#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>
#include <luci/IR/Nodes/CircleFullyConnected.h>
#include <luci/IR/Nodes/CircleConst.h>
#include <luci/IR/Module.h>
#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>

#include <gtest/gtest.h>

using namespace record_hessian;

TEST(RecordHessianTest, profileDataInvalidInputPath_NEG)
{
  // Create a module and a graph
  auto m = luci::make_module();

  // Initialize RecordHessian
  RecordHessian rh;
  rh.initialize(m.get());

  // Provide an invalid input_data_path
  std::string invalid_input_data_path = "invalid_h5_file";

  // Call profileData and expect an exception
  EXPECT_ANY_THROW(
    { std::unique_ptr<HessianMap> hessian_map = rh.profileData(invalid_input_data_path); });
}

TEST(RecordHessianTest, profileDataNonexistingFile_NEG)
{
  // Create a module and a graph
  auto m = luci::make_module();

  // Initialize RecordHessian
  RecordHessian rh;
  rh.initialize(m.get());

  // // Provide an invalid input_data_path
  std::string non_existing_h5 = "non_existing.h5";

  // // Call profileData and expect an exception
  EXPECT_ANY_THROW({ std::unique_ptr<HessianMap> hessian_map = rh.profileData(non_existing_h5); });
}
