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

#ifndef __ONERT_TRAIN_ARGS_H__
#define __ONERT_TRAIN_ARGS_H__

#include <string>
#include <unordered_map>
#include <vector>
#include <boost/program_options.hpp>

#include "types.h"

namespace po = boost::program_options;

namespace onert_train
{

using TensorShapeMap = std::unordered_map<uint32_t, TensorShape>;

#if defined(ONERT_HAVE_HDF5) && ONERT_HAVE_HDF5 == 1
enum class WhenToUseH5Shape
{
  NOT_PROVIDED, // Param not provided
  PREPARE, // read shapes in h5 file and set them as inputs' shape before calling nnfw_prepare()
  RUN,     // read shapes in h5 file and set them as inputs' shape before calling nnfw_run()
};
#endif

class Args
{
public:
  Args(const int argc, char **argv);
  void print(void);

  const std::string &getPackageFilename(void) const { return _package_filename; }
  const std::string &getModelFilename(void) const { return _model_filename; }
  const bool useSingleModel(void) const { return _use_single_model; }
  const int getEpoch(void) const { return _epoch; }
  const int getBatchSize(void) const { return _batch_size; }
  const float getLearningRate(void) const { return _learning_rate; }
  const std::string &getLossFunction(void) const { return _loss_function; }
  const std::string &getOptimizer(void) const { return _optimizer; }
  const bool printVersion(void) const { return _print_version; }
  const int getVerboseLevel(void) const { return _verbose_level; }
  std::unordered_map<uint32_t, uint32_t> getOutputSizes(void) const { return _output_sizes; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  po::positional_options_description _positional;
  po::options_description _options;

  std::string _package_filename;
  std::string _model_filename;
  bool _use_single_model = false;
  int _epoch;
  int _batch_size;
  float _learning_rate;
  std::string _loss_function;
  std::string _optimizer;
  bool _print_version = false;
  int _verbose_level;
  std::unordered_map<uint32_t, uint32_t> _output_sizes;
};

} // end of namespace onert_train

#endif // __ONERT_TRAIN_ARGS_H__
