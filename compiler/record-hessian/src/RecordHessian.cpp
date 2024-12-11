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
#include "record-hessian/HessianObserver.h"

#include <dio_hdf5/HDF5Importer.h>

#include <iostream>

using Shape = std::vector<loco::Dimension>;

namespace record_hessian
{
void RecordHessian::initialize(luci::Module *module)
{
  // Create and initialize interpreters and observers

  _module = module;

  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module);
  auto observer = std::make_unique<HessianObserver>();

  interpreter->attachObserver(observer.get());

  _observer = std::move(observer);
  _interpreter = std::move(interpreter);
}
std::unique_ptr<HessianMap> RecordHessian::profileData(const std::string &input_data_path)
{
  try
  {
    dio::hdf5::HDF5Importer importer(input_data_path);
    // To be implemented
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("RecordHessian: HDF5 error occurred.");
  }

  return getObserver()->hessianData();
}

} // namespace record_hessian
