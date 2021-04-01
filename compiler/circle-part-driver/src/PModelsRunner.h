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

#ifndef __CIRCLE_PRUNNER_PMODELS_RUNNER_H__
#define __CIRCLE_PRUNNER_PMODELS_RUNNER_H__

#include <crew/PConfig.h>

#include <string>
#include <vector>

namespace prunner
{

using RunModel = std::string;

using RunModels = std::vector<RunModel>;

/**
 * @brief PModelsRunner runs partitioned models from input data file and stores
 *        output data to a file
 */
class PModelsRunner
{
public:
  PModelsRunner() = default;

public:
  bool load_config(const std::string &filename);
  void load_inputs(const std::string &input_prefix, int32_t num_inputs);
  bool run(void);
  void save_outputs(const std::string &output_file);

private:
  crew::PConfig _pconfig;
  RunModels _models_to_run;
};

} // namespace prunner

#endif // __CIRCLE_PRUNNER_PMODELS_RUNNER_H__
