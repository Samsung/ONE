/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RecordMinMax.h"
#include "CircleExpContract.h"

#include <luci/Importer.h>
#include <luci/CircleExporter.h>

#include <fstream>
#include <stdexcept>

namespace record_minmax
{

void RecordMinMax::initialize(const std::string &input_model_path)
{
  // Load model from the file
  std::ifstream fs(input_model_path, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + input_model_path + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());
  _module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (_module == nullptr)
  {
    throw std::runtime_error("ERROR: Failed to load '" + input_model_path + "'");
  }

  // TODO: Initialize profiling-supported interpreter
}

void RecordMinMax::profileData(const std::string &input_data_path)
{
  // TODO: Collect min/max data for the given input data (.h5)

  // TODO: Determine the final min/max for each activation
  //       E.g., using clipping, averaging
}

void RecordMinMax::saveModel(const std::string &output_model_path)
{
  // TODO: Write min/max data to activation tensors in CircleNodes

  // Export to output Circle file
  luci::CircleExporter exporter;
  CircleExpContract contract(_module.get(), output_model_path);

  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("ERROR: Failed to export '" + output_model_path + "'");
  }
}

} // namespace record_minmax
