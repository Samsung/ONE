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
  // TODO add implementation
  (void)input_prefix;
  (void)num_inputs;
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
