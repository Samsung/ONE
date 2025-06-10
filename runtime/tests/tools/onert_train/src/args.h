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

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
#include <arser/arser.h>

#include "nnfw_experimental.h"
#include "types.h"

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
  const std::string &getCheckpointFilename(void) const { return _checkpoint_filename; }
  const std::string &getExportCircleFilename(void) const { return _export_circle_filename; }
  const std::string &getExportCirclePlusFilename(void) const { return _export_circleplus_filename; }
  const std::string &getExportCheckpointFilename(void) const { return _export_checkpoint_filename; }
  bool useSingleModel(void) const { return _use_single_model; }
  const std::string &getLoadRawInputFilename(void) const { return _load_raw_input_filename; }
  const std::string &getLoadRawExpectedFilename(void) const { return _load_raw_expected_filename; }
  bool getMemoryPoll(void) const { return _mem_poll; }
  int32_t getEpoch(void) const { return _epoch; }
  const std::optional<int32_t> getBatchSize(void) const { return _batch_size; }
  const std::optional<float> getLearningRate(void) const { return _learning_rate; }
  const std::optional<NNFW_TRAIN_LOSS> getLossType(void) const { return _loss_type; }
  const std::optional<NNFW_TRAIN_LOSS_REDUCTION> getLossReductionType(void) const
  {
    return _loss_reduction_type;
  }
  const std::optional<NNFW_TRAIN_OPTIMIZER> getOptimizerType(void) const { return _optimizer_type; }
  int32_t getMetricType(void) const { return _metric_type; }
  float getValidationSplit(void) const { return _validation_split; }
  bool printVersion(void) const { return _print_version; }
  int32_t getVerboseLevel(void) const { return _verbose_level; }
  std::unordered_map<uint32_t, uint32_t> getOutputSizes(void) const { return _output_sizes; }
  uint32_t num_of_trainable_ops(void) const { return _num_of_trainable_ops; }

private:
  void Initialize();
  void Parse(const int argc, char **argv);

private:
  // supported loss list and it's default value
  const std::vector<NNFW_TRAIN_LOSS> valid_loss = {
    NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR,
    NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY,
  };

  // supported loss reduction type list and it's default value
  const std::vector<NNFW_TRAIN_LOSS_REDUCTION> valid_loss_rdt = {
    NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE,
    NNFW_TRAIN_LOSS_REDUCTION_SUM,
  };

  // supported optimizer list and it's default value
  const std::vector<NNFW_TRAIN_OPTIMIZER> valid_optim = {
    NNFW_TRAIN_OPTIMIZER_SGD,
    NNFW_TRAIN_OPTIMIZER_ADAM,
  };

private:
  arser::Arser _arser;

  std::string _package_filename;
  std::string _model_filename;
  std::string _checkpoint_filename;
  std::string _export_circle_filename;
  std::string _export_circleplus_filename;
  std::string _export_checkpoint_filename;
  bool _use_single_model = false;
  std::string _load_raw_input_filename;
  std::string _load_raw_expected_filename;
  bool _mem_poll;
  int32_t _epoch;
  std::optional<int32_t> _batch_size;
  std::optional<float> _learning_rate;
  std::optional<NNFW_TRAIN_LOSS> _loss_type;
  std::optional<NNFW_TRAIN_LOSS_REDUCTION> _loss_reduction_type;
  std::optional<NNFW_TRAIN_OPTIMIZER> _optimizer_type;
  int32_t _metric_type;
  float _validation_split;
  bool _print_version = false;
  int32_t _verbose_level;
  std::unordered_map<uint32_t, uint32_t> _output_sizes;
  int32_t _num_of_trainable_ops;
};

} // end of namespace onert_train

#endif // __ONERT_TRAIN_ARGS_H__
