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

#include "OMTrainingInterpreter.h"
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#define MODEL_TYPE float

#define CLASSIFICATION_TASK 0

namespace
{

using DataBuffer = std::vector<char>;

void readDataFromFile(const std::string &filename, char *data, size_t data_size,
                      size_t start_position = 0)
{
  std::streampos start = start_position;

  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");

  fs.seekg(start);

  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
  fs.close();
}

void readDataFromFile(std::ifstream &fs, const std::string &filename, char *data, size_t data_size,
                      size_t start_position = 0)
{
  std::streampos start = start_position;

  fs.seekg(start);

  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
}

void writeDataToFile(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

DataBuffer readFile(const char *path)
{
  std::ifstream file(path, std::ios::binary | std::ios::in);
  if (!file.good())
  {
    std::string errmsg = "Failed to open file";
    throw std::runtime_error(errmsg.c_str());
  }

  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // reserve capacity
  DataBuffer model_data(fileSize);

  // read the data
  file.read(model_data.data(), fileSize);
  if (file.fail())
  {
    std::string errmsg = "Failed to read file";
    throw std::runtime_error(errmsg.c_str());
  }

  return model_data;
}

} // namespace

/*
 * @brief EvalDriver main
 *
 *  Driver for testing training onert micro
 *  Current example for testing classification task
 *
 */
int entry(int argc, char **argv)
{
  if (argc != 11 and argc != 10)
  {
    std::cerr << "Two variant of usage with and without wof file: " << argv[0]
              << " <path/to/circle/model> "
                 " optional(<path/to/wof/file>) <path/to/save/trained/model> "
                 "<path/to/save/checkpoint> <path/to/input/train_data> "
                 "<path/to/input/target_train_data> "
                 "<path/to/input/test_data> <path/to/input/target_test_data> num_of_train_smpl "
                 "num_of_test_smpl\n";
    return EXIT_FAILURE;
  }

  const char *circle_model_path = nullptr;
  const char *wof_file_path = nullptr;
  const char *output_trained_file_path = nullptr;
  const char *input_input_train_data_path = nullptr;
  const char *input_target_train_data_path = nullptr;
  const char *input_input_test_data_path = nullptr;
  const char *input_target_test_data_path = nullptr;
  const char *checkpoints_path = nullptr;
  int32_t num_train_data_samples = 0;
  int32_t num_test_data_samples = 0;

  if (argc == 11)
  {
    circle_model_path = argv[1];
    wof_file_path = argv[2];
    output_trained_file_path = argv[3];
    checkpoints_path = argv[4];
    input_input_train_data_path = argv[5];
    input_target_train_data_path = argv[6];
    input_input_test_data_path = argv[7];
    input_target_test_data_path = argv[8];
    num_train_data_samples = atoi(argv[9]);
    num_test_data_samples = atoi(argv[10]);
  }
  else if (argc == 10)
  {
    circle_model_path = argv[1];
    output_trained_file_path = argv[2];
    checkpoints_path = argv[3];
    input_input_train_data_path = argv[4];
    input_target_train_data_path = argv[5];
    input_input_test_data_path = argv[6];
    input_target_test_data_path = argv[7];
    num_train_data_samples = atoi(argv[8]);
    num_test_data_samples = atoi(argv[9]);
  }
  else
  {
    throw std::runtime_error("Unknown commands number\n");
  }

  DataBuffer circle_model = readFile(circle_model_path);
  DataBuffer wof_data;
  // If defined wof file
  if (wof_file_path != nullptr)
    wof_data = readFile(wof_file_path);

  // Configure training mode
  onert_micro::OMConfig config;

  // Save model size and model ptr in config
  config.model_size = circle_model.size();
  config.model_ptr = circle_model.data();

  // If defined wof file
  if (wof_file_path != nullptr)
    config.wof_ptr = nullptr;

  // Set user defined training settings
  const uint32_t training_epochs = 30;
  const float lambda = 0.1f;
  const uint32_t BATCH_SIZE = 32;
  const uint32_t INPUT_SIZE = 13;
  const uint32_t OUTPUT_SIZE = 1;
  const uint32_t num_train_layers = 0;
  const onert_micro::OMLoss loss = onert_micro::MSE;
  const onert_micro::OMTrainOptimizer train_optim = onert_micro::ADAM;
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
    train_context.optimizer = train_optim;
    train_context.beta = beta;
    train_context.beta_squares = beta_squares;
    train_context.epsilon = epsilon;
    train_context.epochs = training_epochs;

    config.training_context = train_context;
  }

  // Create training interpreter and import models
  onert_micro::OMTrainingInterpreter train_interpreter;
  train_interpreter.importTrainModel(circle_model.data(), config);

  // Temporary buffer to read input data from file using BATCH_SIZE
  float training_input[BATCH_SIZE * INPUT_SIZE];
  float training_target[BATCH_SIZE * OUTPUT_SIZE];
  // Note: here test size used with BATCH_SIZE = 1
  float test_input[INPUT_SIZE];
  float test_target[OUTPUT_SIZE];

  std::vector<float> accuracy_v;
  std::vector<float> cross_entropy_v;
  std::vector<float> mse_v;
  std::vector<float> mae_v;

  float max_accuracy = std::numeric_limits<float>::min();
  float min_mae = std::numeric_limits<float>::max();

  for (uint32_t e = 0; e < training_epochs; ++e)
  {
    // Run train for current epoch
    std::cout << "Run training for epoch: " << e + 1 << "/" << training_epochs << "\n";
    config.training_context.num_epoch = e + 1;
    uint32_t num_steps = num_train_data_samples / BATCH_SIZE;
    for (int i = 0; i < num_steps; ++i)
    {
      uint32_t cur_batch_size = std::min(BATCH_SIZE, num_train_data_samples - BATCH_SIZE * i - 1);
      cur_batch_size = std::max(1u, cur_batch_size);

      config.training_context.batch_size = cur_batch_size;

      // Read current input and target data
      readDataFromFile(input_input_train_data_path, reinterpret_cast<char *>(training_input),
                       sizeof(float) * INPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * INPUT_SIZE * BATCH_SIZE);

      readDataFromFile(input_target_train_data_path, reinterpret_cast<char *>(training_target),
                       sizeof(float) * OUTPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * OUTPUT_SIZE * BATCH_SIZE);

      // Set input and target
      train_interpreter.setInput(reinterpret_cast<uint8_t *>(training_input), 0);
      train_interpreter.setTarget(reinterpret_cast<uint8_t *>(training_target), 0);

      // Train with current batch size
      train_interpreter.trainSingleStep(config);

      float mse = 0.f;
      float mae = 0.f;
      float cross_entropy_metric = 0.f;
      float accuracy = 0.f;

      if (CLASSIFICATION_TASK)
      {
        // Evaluate cross_entropy and accuracy metrics
        train_interpreter.evaluateMetric(onert_micro::CROSS_ENTROPY_METRICS,
                                         reinterpret_cast<void *>(&cross_entropy_metric),
                                         cur_batch_size);
        train_interpreter.evaluateMetric(onert_micro::ACCURACY, reinterpret_cast<void *>(&accuracy),
                                         cur_batch_size);

        // Save them into vectors
        accuracy_v.push_back(accuracy);
        cross_entropy_v.push_back(cross_entropy_metric);
      }
      else
      {
        // Evaluate mse and mae metrics
        train_interpreter.evaluateMetric(onert_micro::MSE_METRICS, reinterpret_cast<void *>(&mse),
                                         cur_batch_size);
        train_interpreter.evaluateMetric(onert_micro::MAE_METRICS, reinterpret_cast<void *>(&mae),
                                         cur_batch_size);

        // Save them into vectors
        mse_v.push_back(mse);
        mae_v.push_back(mae);
      }
    }

    // Reset num step value
    config.training_context.num_step = 0;

    // Calculate and print average values
    if (CLASSIFICATION_TASK)
    {
      float sum_acc = std::accumulate(accuracy_v.begin(), accuracy_v.end(), 0.f);
      float sum_ent = std::accumulate(cross_entropy_v.begin(), cross_entropy_v.end(), 0.f);
      std::cout << "Train Average ACCURACY = " << sum_acc / accuracy_v.size() << "\n";
      std::cout << "Train Average CROSS ENTROPY = " << sum_ent / cross_entropy_v.size() << "\n";
    }
    else
    {
      float sum_mse = std::accumulate(mse_v.begin(), mse_v.end(), 0.f);
      float sum_mae = std::accumulate(mae_v.begin(), mae_v.end(), 0.f);
      std::cout << "Train Average MSE = " << sum_mse / mse_v.size() << "\n";
      std::cout << "Train Average MAE = " << sum_mae / mae_v.size() << "\n";
    }

    // Run test for current epoch
    std::cout << "Run test for epoch: " << e + 1 << "/" << training_epochs << "\n";
    num_steps = num_test_data_samples;

    accuracy_v.clear();
    cross_entropy_v.clear();

    for (int i = 0; i < num_steps; ++i)
    {
      uint32_t cur_batch_size = 1;
      readDataFromFile(input_input_test_data_path, reinterpret_cast<char *>(test_input),
                       sizeof(float) * INPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * INPUT_SIZE);

      readDataFromFile(input_target_test_data_path, reinterpret_cast<char *>(test_target),
                       sizeof(float) * OUTPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * OUTPUT_SIZE);

      train_interpreter.setInput(reinterpret_cast<uint8_t *>(test_input), 0);
      train_interpreter.setTarget(reinterpret_cast<uint8_t *>(test_target), 0);

      float mse = 0.f;
      float mae = 0.f;
      float cross_entropy_metric = 0.f;
      float accuracy = 0.f;

      if (CLASSIFICATION_TASK)
      {
        // Evaluate cross_entropy and accuracy metrics
        train_interpreter.evaluateMetric(onert_micro::CROSS_ENTROPY_METRICS,
                                         reinterpret_cast<void *>(&cross_entropy_metric),
                                         cur_batch_size);
        train_interpreter.evaluateMetric(onert_micro::ACCURACY, reinterpret_cast<void *>(&accuracy),
                                         cur_batch_size);

        // Save them into vectors
        accuracy_v.push_back(accuracy);
        cross_entropy_v.push_back(cross_entropy_metric);
      }
      else
      {
        // Evaluate mse and mae metrics
        train_interpreter.evaluateMetric(onert_micro::MSE_METRICS, reinterpret_cast<void *>(&mse),
                                         cur_batch_size);
        train_interpreter.evaluateMetric(onert_micro::MAE_METRICS, reinterpret_cast<void *>(&mae),
                                         cur_batch_size);

        // Save them into vectors
        mse_v.push_back(mse);
        mae_v.push_back(mae);
      }
    }
    // Calculate and print average values
    if (CLASSIFICATION_TASK)
    {
      float sum_acc = std::accumulate(accuracy_v.begin(), accuracy_v.end(), 0.f);
      float sum_ent = std::accumulate(cross_entropy_v.begin(), cross_entropy_v.end(), 0.f);
      std::cout << "Test Average ACCURACY = " << sum_acc / accuracy_v.size() << "\n";
      std::cout << "Test Average CROSS ENTROPY = " << sum_ent / cross_entropy_v.size() << "\n";

      // Best checkpoint part
      float acc = sum_acc / accuracy_v.size();
      if (acc > max_accuracy)
      {
        // Save best checkpoint
        train_interpreter.saveCheckpoint(config, checkpoints_path);
        max_accuracy = acc;
        std::cout << "Found new max Test ACCURACY = " << max_accuracy << " in epoch = " << e + 1
                  << " / " << training_epochs << "\n";
      }
    }
    else
    {
      float sum_mse = std::accumulate(mse_v.begin(), mse_v.end(), 0.f);
      float sum_mae = std::accumulate(mae_v.begin(), mae_v.end(), 0.f);
      std::cout << "Test Average MSE = " << sum_mse / mse_v.size() << "\n";
      std::cout << "Test Average MAE = " << sum_mae / mae_v.size() << "\n";

      // Best checkpoint part
      float acc = sum_mae / mae_v.size();
      if (acc < min_mae)
      {
        // Save best checkpoint
        train_interpreter.saveCheckpoint(config, checkpoints_path);
        min_mae = acc;
        std::cout << "Found new min Test MAE = " << min_mae << " in epoch = " << e + 1 << " / "
                  << training_epochs << "\n";
      }
    }
  }

  // Load best model
  train_interpreter.loadCheckpoint(config, checkpoints_path);

  // Save training best result
  train_interpreter.saveModel(config, output_trained_file_path);

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
