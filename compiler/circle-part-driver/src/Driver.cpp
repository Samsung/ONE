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

#include <luci/Log.h>
#include <cstdlib>
#include <iostream>

int entry(int argc, char **argv)
{
  LOGGER(l);

  if (argc != 5)
  {
    std::cerr
      << "Usage: " << argv[0]
      << " <path/to/partition/config> <num_inputs> <path/to/input/prefix> <path/to/output/file>\n";
    return EXIT_FAILURE;
  }
  // NOTE: about input/output data file name
  // - I/O file name format is like filename.ext0, filename.ext1, ...
  // NOTE: about output shape
  // - file name with filename.ext0.shape, filename.ext1.shape, ...
  //   having one line text content of CSV format(like H,W or N,C,H,W)

  const char *config_filename = argv[1];
  const int32_t num_inputs = atoi(argv[2]);
  const char *input_prefix = argv[3];
  const char *output_file = argv[4];

  prunner::PModelsRunner pmrunner;

  INFO(l) << "Read config file: " << config_filename << std::endl;
  if (not pmrunner.load_config(config_filename))
    return EXIT_FAILURE;

  INFO(l) << "Read input file: " << input_prefix << ", #inputs: " << num_inputs << std::endl;
  pmrunner.load_inputs(input_prefix, num_inputs);

  INFO(l) << "Run all partitioned models..." << std::endl;
  if (!pmrunner.run())
    return EXIT_FAILURE;

  INFO(l) << "Save output file: " << output_file << std::endl;
  pmrunner.save_outputs(output_file);

  return EXIT_SUCCESS;
}
