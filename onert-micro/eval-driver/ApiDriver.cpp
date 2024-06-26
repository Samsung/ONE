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

#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

#include "onert-micro.h"

#define MODEL_TYPE float

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

bool is_correct(const uint32_t flat_size, float *calculated_data, float *target_data)
{
  // Find target class
  float target_class = 0.f;
  float target_max_val = target_data[0];
  for (uint32_t i = 0; i < flat_size; ++i)
  {
    if (target_max_val < target_data[i])
    {
      target_max_val = target_data[i];
      target_class = static_cast<float>(i);
    }
  }
  // Find predicted class
  float pred_class = 0.f;
  float pred_max_val = calculated_data[0];
  for (uint32_t i = 0; i < flat_size; ++i)
  {
    if (pred_max_val < calculated_data[i])
    {
      pred_max_val = calculated_data[i];
      pred_class = static_cast<float>(i);
    }
  }

  return pred_class == target_class ? true : false;
}

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

  // Set user defined training settings
  const uint32_t training_epochs = 20;
  const float learning_rate = 0.001f;
  const uint32_t BATCH_SIZE = 32;
  const uint32_t INPUT_SIZE = 180;
  const uint32_t OUTPUT_SIZE = 4;
  const uint32_t num_train_layers = 10;
  const float beta = 0.9;
  const float beta_squares = 0.999;
  const float epsilon = 1e-07;

  nnfw_session *session;
  nnfw_create_session(&session);
  nnfw_load_model_from_file(session, circle_model_path);
  nnfw_train_prepare(session);

  // Temporary buffer to read input data from file using BATCH_SIZE
  float training_input[BATCH_SIZE * INPUT_SIZE];
  float training_target[BATCH_SIZE * OUTPUT_SIZE];
  // Note: here test size used with BATCH_SIZE = 1
  float test_input[INPUT_SIZE];
  float test_target[OUTPUT_SIZE];
  std::vector<float> cross_entropy_v2;

  float max_accuracy = 0;

  for (uint32_t e = 0; e < training_epochs; ++e)
  {
    // Run train for current epoch
    std::cout << "Run training for epoch: " << e + 1 << "/" << training_epochs << "\n";

    uint32_t num_steps = num_train_data_samples / BATCH_SIZE;
    for (int i = 0; i < num_steps; ++i)
    {
      uint32_t cur_batch_size = std::min(BATCH_SIZE, num_train_data_samples - BATCH_SIZE * i);
      cur_batch_size = std::max(1u, cur_batch_size);

      // Read current input and target data
      readDataFromFile(input_input_train_data_path, reinterpret_cast<char *>(training_input),
                       sizeof(float) * INPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * INPUT_SIZE * BATCH_SIZE);

      readDataFromFile(input_target_train_data_path, reinterpret_cast<char *>(training_target),
                       sizeof(float) * OUTPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * OUTPUT_SIZE * BATCH_SIZE);

      // Set input and target
      nnfw_tensorinfo ti = {.dtype = NNFW_TYPE_TENSOR_FLOAT32, .rank = 2, .dims = {1, 180}};
      nnfw_train_set_input(session, 0, training_input, &ti);
      nnfw_train_set_expected(session, 0, training_target, nullptr);

      // Train with current batch size
      nnfw_train(session, true);

      float cross_entropy_metric = 0.f;

      std::cout << "step " << i << "\n";
      // Evaluate cross_entropy and accuracy metrics
      nnfw_train_get_loss(session, 0, &cross_entropy_metric);
      std::cout << "Train CROSS ENTROPY = " << cross_entropy_metric << "\n";
      cross_entropy_v2.push_back(cross_entropy_metric);
    }
    // Calculate and print average values
    float sum_ent = std::accumulate(cross_entropy_v2.begin(), cross_entropy_v2.end(), 0.f);
    std::cout << "Train Average CROSS ENTROPY for 2 = " << sum_ent / cross_entropy_v2.size()
              << "\n";
    cross_entropy_v2.clear();

    // Run test for current epoch
    std::cout << "Run test for epoch: " << e + 1 << "/" << training_epochs << "\n";
    num_steps = num_train_data_samples;
    int correct_predictions = 0;
    for (int i = 0; i < num_steps; ++i)
    {
      uint32_t cur_batch_size = 1;
      readDataFromFile(input_input_train_data_path, reinterpret_cast<char *>(test_input),
                       sizeof(float) * INPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * INPUT_SIZE);

      readDataFromFile(input_target_train_data_path, reinterpret_cast<char *>(test_target),
                       sizeof(float) * OUTPUT_SIZE * cur_batch_size,
                       i * sizeof(MODEL_TYPE) * OUTPUT_SIZE);

      nnfw_tensorinfo ti = {.dtype = NNFW_TYPE_TENSOR_FLOAT32, .rank = 2, .dims = {1, 180}};
      nnfw_train_set_input(session, 0, test_input, &ti);
      nnfw_train_set_expected(session, 0, test_target, nullptr);

      float output[4];
      nnfw_train_set_output(session, 0, NNFW_TYPE_TENSOR_FLOAT32, output, 4);
      nnfw_train(session, false);
      correct_predictions += (is_correct(4, output, test_target)) ? 1 : 0;
    }
    // Calculate and print accuracy
    float accuracy = (float)correct_predictions / num_train_data_samples;
    printf("Accuracy: %f\n", accuracy);
    if (accuracy > max_accuracy)
    {
      // Save best checkpoint
      // train_interpreter.saveCheckpoint(config, checkpoints_path);
      nnfw_train_export_checkpoint(session, checkpoints_path);
      max_accuracy = accuracy;
      std::cout << "Found better accuracy = " << max_accuracy << " in epoch = " << e + 1 << " / "
                << training_epochs << "\n";
    }
  }

  // Load best model
  nnfw_train_import_checkpoint(session, checkpoints_path);

  // Save final result
  nnfw_train_export_circle(session, output_trained_file_path);

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
