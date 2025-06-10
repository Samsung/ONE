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

#include "include/SparseBackpropagationHandler.h"
#include "include/TensorRankSparseBackpropagationHandler.h"

#include "TrainingDriverHandler.h"

#include <iostream>

int entry(int argc, char **argv)
{
  if (argc != 9 and argc != 10)
  {
    std::cerr << "Two variant of usage with and without wof file: " << argv[0]
              << " <path/to/circle/model> "
                 " optional(<path/to/wof/file>) <path/to/save/train/config/result> "
                 "<path/to/input/train_data> "
                 "<path/to/input/target_train_data> "
                 "<path/to/input/test_data> "
                 "<path/to/input/target_test_data>"
                 "num_of_train_smpl "
                 "num_of_test_smpl\n";
    return EXIT_FAILURE;
  }

  training_configure_tool::TrainData train_data;

  if (argc == 10)
  {
    train_data.circle_model_path = argv[1];
    train_data.wof_file_path = argv[2];
    train_data.output_tool_file_path = argv[3];
    train_data.input_input_train_data_path = argv[4];
    train_data.input_target_train_data_path = argv[5];
    train_data.input_input_test_data_path = argv[6];
    train_data.input_target_test_data_path = argv[7];
    train_data.num_train_data_samples = atoi(argv[8]);
    train_data.num_test_data_samples = atoi(argv[9]);
  }
  else if (argc == 9)
  {
    train_data.circle_model_path = argv[1];
    train_data.output_tool_file_path = argv[2];
    train_data.input_input_train_data_path = argv[3];
    train_data.input_target_train_data_path = argv[4];
    train_data.input_input_test_data_path = argv[5];
    train_data.input_target_test_data_path = argv[6];
    train_data.num_train_data_samples = atoi(argv[7]);
    train_data.num_test_data_samples = atoi(argv[8]);
  }
  else
  {
    throw std::runtime_error("Unknown commands number\n");
  }

  // Configure training mode
  onert_micro::OMConfig config;

  // Set user defined training settings
  const uint32_t training_epochs = 25;
  const float lambda = 0.001f;
  const uint32_t BATCH_SIZE = 64;
  const uint32_t num_train_layers = 0;
  const onert_micro::OMLoss loss = onert_micro::CROSS_ENTROPY;
  const onert_micro::OMTrainOptimizer train_optimizer = onert_micro::ADAM;
  const float beta = 0.9;
  const float beta_squares = 0.999;
  const float epsilon = 1e-07;

  config.train_mode = true;
  {
    onert_micro::OMTrainingContext train_context;
    train_context.batch_size = BATCH_SIZE;
    train_context.num_of_train_layers = num_train_layers;
    train_context.learning_rate = lambda;
    train_context.loss = loss;
    train_context.optimizer = train_optimizer;
    train_context.beta = beta;
    train_context.beta_squares = beta_squares;
    train_context.epsilon = epsilon;
    train_context.epochs = training_epochs;

    config.training_context = train_context;
  }

  train_data.metrics_to_check_best_config = onert_micro::CROSS_ENTROPY_METRICS;
  train_data.memory_above_restriction = 300000;
  train_data.acceptable_diff = 0.02;
  // Find sparse backpropagation best configure
  std::unordered_set<uint16_t> best_trainable_op_indexes;
  training_configure_tool::findBestTrainableOpIndexes(config, train_data,
                                                      best_trainable_op_indexes);

  // Find the best train tensors ranks
  training_configure_tool::TrainConfigFileData config_result;
  auto res = training_configure_tool::findBestSparseBackpropagationTensorsRanks(
    config, train_data, best_trainable_op_indexes, config_result.trainable_op_indexes_with_ranks);

  // Save result into file
  assert(!config_result.trainable_op_indexes_with_ranks.empty());
  training_configure_tool::createResultFile(config_result, train_data.output_tool_file_path);

  return EXIT_SUCCESS;
}

int entry(int argc, char **argv);

#ifdef NDEBUG
int main(int argc, char **argv)
{
  try
  {
    return entry(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
  }

  return 255;
}
#else  // NDEBUG
int main(int argc, char **argv)
{
  // NOTE main does not catch internal exceptions for debug build to make it easy to
  //      check the stacktrace with a debugger
  return entry(argc, argv);
}
#endif // !NDEBUG
