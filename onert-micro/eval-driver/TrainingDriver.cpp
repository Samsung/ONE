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
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <math.h>
#include <algorithm>

#define MODEL_TYPE float

namespace
{

using DataBuffer = std::vector<char>;

void readDataFromFile(const std::string &filename, char *data, size_t data_size, size_t start_position = 0)
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

void readDataFromFile(std::ifstream  &fs, const std::string &filename, char *data, size_t data_size, size_t start_position = 0)
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

void printPredAndTargetsValues(float *pred, float *target, size_t size)
{
  std::cout << "Print Valus: PREDICTED TARGET DIFF\n";
  for (uint32_t i = 0; i < size; ++i)
  {
    std::cout << pred[i] << "  " << target[i] << "  " << target[i] - pred[i] << "\n";
  }
}

float calculateMSE(float *pred, float *target, size_t size)
{
  float mse = 0;
  for (uint32_t i = 0; i < size; ++i)
  {
    mse += (pred[i] - target[i]) * (pred[i] - target[i]);
  }
  std::cout << "Calculated MSE = " << mse / size << "\n";
  return mse;
}

float calculateMAE(float *pred, float *target, size_t size)
{
  float mae = 0;
  for (uint32_t i = 0; i < size; ++i)
  {
    mae += std::abs(pred[i] - target[i]);
  }
  std::cout << "Calculated MAE = " << mae / size << "\n";
  return mae;
}

float calculateCrossEntropy(float *pred, float *target, size_t size)
{
  float cce = 0;
  for (uint32_t i = 0; i < size; ++i)
  {
    cce += std::log(pred[i]) * target[i];
  }
  std::cout << "Calculated CROSS_ENTROPY = " << cce << "\n";
  return -cce;
}

int predicted_class(float *pred, size_t size)
{
  int max = 0;
  float max_value = pred[0];
  for (uint32_t i = 1; i < size; ++i)
  {
    if (max_value < pred[i])
    {
      max = i;
      max_value = pred[i];
    }
  }

  //std::cout << "Predicted class = " << max << "\n";
  return max;
}

void measure_accuracy(std::vector<int> &predicted_labels, std::vector<int> &targets_labels)
{
  assert(predicted_labels.size() == targets_labels.size());
  int corrected_num = 0;
  for (uint32_t i = 0; i < predicted_labels.size(); ++i)
  {
    int pred_value = predicted_labels[i];
    int real_value = targets_labels[i];
    if (pred_value == real_value)
      corrected_num++;
  }

  std::cout << "Calculated accuracy = " << static_cast<float>(corrected_num) / static_cast<float>(predicted_labels.size()) << "\n";
}

void printDataVector(float *data, int num_size, int count)
{
  std::cout << "Print data \n";
  for (int k = 0; k < num_size; ++k)
  {
    std::cout << data[k] << " ";
  }
  std::cout << "\n";
}

} // namespace

/*
 * @brief EvalDriver main
 *
 *        Driver for testing training onert micro
 *
 */
int entry(int argc, char **argv)
{
  if (argc != 11)
  {
    std::cerr
      << "Usage: " << argv[0]
      << " <path/to/circle/model/without/weights> <path/to/circle/backpropagation/model>"
         " <path/to/wof/file> <path/to/save/wof/file> <path/to/input/train_data> <path/to/input/label_train_data> "
         "<path/to/input/test_data> <path/to/input/label_test_data> num_of_train_smpl "
         "num_of_test_smpl\n";
    return EXIT_FAILURE;
  }

  const char *circle_model_path = argv[1];
  const char *backprop_model_path = argv[2];
  const char *wof_file_path = argv[3];
  const char *output_wof_file_path = argv[4];
  const char *input_input_train_data_path = argv[5];
  const char *input_target_train_data_path = argv[6];
  const char *input_input_test_data_path = argv[7];
  const char *input_target_test_data_path = argv[8];
  const int32_t num_train_data_samples = atoi(argv[9]);
  const int32_t num_test_data_samples = atoi(argv[10]);

  DataBuffer circle_model = readFile(circle_model_path);
  DataBuffer backprop_model = readFile(backprop_model_path);
  DataBuffer wof_data = readFile(wof_file_path);

  // Set user defined training settings
  const uint32_t training_epochs = 20;
  const float lambda = 0.001f;

  // Configure training mode
  onert_micro::OMConfig config;
  config.wof_ptr = wof_data.data();
  config.train_mode = true;
  {
    onert_micro::OMTrainingConfig trainConfig;
    trainConfig.lambda = lambda;
    trainConfig.optimization_strategy = onert_micro::ADAM;
    trainConfig.beta_squares = 0.999f;
    trainConfig.beta = 0.9f;
    trainConfig.batches = 150;

    config.train_config = trainConfig;
  }

  // Create training interpreter and import models
  onert_micro::OMTrainingInterpreter train_interpreter;
  train_interpreter.import(circle_model.data(), backprop_model.data(), config);

  const auto model_inputs_num = train_interpreter.getNumberOfInputs();
  const auto model_outputs_num = train_interpreter.getNumberOfOutputs();
  const auto model_targets_num = train_interpreter.getNumberOfTargets();

  if (model_outputs_num != model_targets_num)
    throw std::runtime_error("Output model size and target size have to be equal");

  assert(model_inputs_num == 1);
  assert(model_outputs_num == 1);
  assert(model_targets_num == 1);

  // Save model input/output/target size
  const auto input_size = train_interpreter.getInputSizeAt(0);
  const auto output_size = train_interpreter.getOutputSizeAt(0);
  const auto target_size = train_interpreter.getTargetSizeAt(0);

  // train data
  std::unique_ptr<float []> train_input_data(new float [input_size]);
  std::unique_ptr<float []> train_target_data(new float [target_size]);

  // Test data
  std::unique_ptr<float []> test_input_data(new float [input_size]);
  std::unique_ptr<float []> test_target_data(new float [target_size]);

  for (uint32_t e = 0; e < training_epochs; ++e)
  {
    train_interpreter.set_training_mode(true);
    std::cout << "Run training for epoch: " << e + 1 << "/" << training_epochs << "\n";
    std::vector<int> predicted_labels;
    std::vector<int> targets_labels;
    for (int i = 0; i < num_train_data_samples; i += config.train_config.batches)
    {
      for (int batch = 0; batch < config.train_config.batches and i + batch < num_train_data_samples; ++batch)
      {
        readDataFromFile(input_input_train_data_path, reinterpret_cast<char *>(train_input_data.get()),
                         sizeof(MODEL_TYPE) * input_size, sizeof(MODEL_TYPE) * input_size * (i + batch));

        readDataFromFile(input_target_train_data_path, reinterpret_cast<char *>(train_target_data.get()),
                         sizeof(MODEL_TYPE) * target_size, sizeof(MODEL_TYPE) * target_size * (i + batch));

  //   printDataVector(train_input_data.get(), input_size, i + batch);

        train_interpreter.allocateInputs();
        // Copy input data
        auto *cur_train_data = train_input_data.get();
        auto cur_input_data = train_interpreter.getInputDataAt(0);
        std::memcpy(cur_input_data, cur_train_data, sizeof(MODEL_TYPE) * input_size);

        train_interpreter.forward();
        train_interpreter.allocateTargets();

        // Copy targets values
        auto *cur_train_target_data = train_target_data.get();
        auto cur_target_data = train_interpreter.getTargetDataAt(0);
        std::memcpy(cur_target_data, cur_train_target_data,
                    sizeof(MODEL_TYPE) * target_size);

        predicted_labels.push_back(predicted_class(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)), target_size));
        targets_labels.push_back(predicted_class(reinterpret_cast<float *>(cur_train_target_data), target_size));

#if 0
//        calculateCrossEntropy(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
//                              reinterpret_cast<float *>(cur_train_target_data),
//                              target_size);
        printPredAndTargetsValues(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
                                  reinterpret_cast<float *>(cur_train_target_data),
                                  target_size);
#endif
        train_interpreter.backward();
      }
      train_interpreter.updateWeights();
      train_interpreter.reset();
    }
    std::cout << "Training accuracy: ";
    measure_accuracy(predicted_labels, targets_labels);

    predicted_labels.clear();
    targets_labels.clear();

    // Run test dataset
    std::vector<float> mse_vector;
    std::vector<float> mae_vector;

    train_interpreter.set_training_mode(false);

    std::cout << "Run test dataset for epoch: " << e + 1 << "/" << training_epochs << "\n";
    for (int i = 0; i < num_test_data_samples; i++)
    {
      // read data
      readDataFromFile(input_input_test_data_path, reinterpret_cast<char *>(test_input_data.get()),
                       sizeof(MODEL_TYPE) * input_size, sizeof(MODEL_TYPE) * input_size * i);

      readDataFromFile(input_target_test_data_path, reinterpret_cast<char *>(test_target_data.get()),
                       sizeof(MODEL_TYPE) * target_size, sizeof(MODEL_TYPE) * target_size * i);

      train_interpreter.allocateInputs();
      // Copy input data
      auto *cur_test_data = test_input_data.get();
      auto cur_input_data = train_interpreter.getInputDataAt(0);
      std::memcpy(cur_input_data, cur_test_data, sizeof(MODEL_TYPE) * input_size);
      train_interpreter.forward();

      predicted_labels.push_back(predicted_class(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)), target_size));
      targets_labels.push_back(predicted_class(test_target_data.get(), target_size));

      train_interpreter.reset();
    }
    std::cout << "Test accuracy: ";
    measure_accuracy(predicted_labels, targets_labels);
  }
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
