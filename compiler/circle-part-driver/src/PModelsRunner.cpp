/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "PModelsRunner.h"

#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>
#include <luci/Log.h>

#include <foder/FileLoader.h>
#include <crew/PConfig.h>

#include <iostream>
#include <vector>
#include <string>

namespace prunner
{

bool PModelsRunner::load_config(const std::string &filename)
{
  if (!crew::read_ini(filename, _pconfig))
  {
    std::cerr << "ERROR: Invalid config ini file: '" << filename << "'" << std::endl;
    return false;
  }

  for (auto &part : _pconfig.parts)
  {
    _models_to_run.push_back(part.model_file);
  }
  return true;
}

void PModelsRunner::load_inputs(const std::string &input_prefix, int32_t num_inputs)
{
  LOGGER(l);

  auto its = _pconfig.source.inputs.begin();
  for (int32_t i = 0; i < num_inputs; ++i, ++its)
  {
    std::string filename = input_prefix + std::to_string(i);

    INFO(l) << "Load input data: " << filename << std::endl;
    foder::FileLoader file_loader{filename};

    std::string input_name = *its;
    _data_stage[input_name] = file_loader.load();

    INFO(l) << "Input: [" << input_name << "], size " << _data_stage[input_name].size()
            << std::endl;
  }
}

bool PModelsRunner::run(void)
{
  // TODO add implementation
  return true;
}

void PModelsRunner::save_outputs(const std::string &output_file)
{
  // TODO add implementation
  (void)output_file;
}

} // namespace prunner
