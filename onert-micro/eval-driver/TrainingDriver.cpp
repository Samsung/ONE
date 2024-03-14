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

#define MODEL_TYPE float

namespace
{

using DataBuffer = std::vector<char>;

void readDataFromFile(const std::string &filename, char *data, size_t data_size)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
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

//void printDataVector(float *data, int num_samples, int num_size)
//{
//  for (int i = 0; i < num_samples; i++)
//  {
//    std::cout << "Cur sample № = " << i + 1 << "\n";
//    for (int j = 0; j < num_inputs; ++j)
//    {
//      for (int k = 0; k > num_size; ++k)
//      {
//        std::cout << data[k + j * num_size + i * num_size * num_inputs];
//      }
//    }
//  }
//}

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
  const char *input_train_data_path = argv[5];
  const char *input_label_train_data_path = argv[6];
  const char *input_test_data_path = argv[7];
  const char *input_label_test_data_path = argv[8];
  const int32_t num_train_data_samples = atoi(argv[9]);
  const int32_t num_test_data_samples = atoi(argv[10]);

  DataBuffer circle_model = readFile(circle_model_path);
  DataBuffer backprop_model = readFile(backprop_model_path);
  DataBuffer wof_data = readFile(wof_file_path);

  // Set user defined training settings
  const uint32_t training_epochs = 100;
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
    trainConfig.batches = 32;

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

  auto train_input_data_raw = std::make_unique<char []>(sizeof(MODEL_TYPE) * input_size * num_train_data_samples);
  auto train_target_data_raw = std::make_unique<char []>(sizeof(MODEL_TYPE) * target_size * num_train_data_samples);

  auto test_input_data_raw = std::make_unique<char []>(sizeof(MODEL_TYPE) * input_size * num_test_data_samples);
  auto test_target_data_raw = std::make_unique<char []>(sizeof(MODEL_TYPE) * target_size * num_test_data_samples);

  // Read train and test inputs
  readDataFromFile(input_train_data_path, train_input_data_raw.get(), sizeof(MODEL_TYPE) * input_size * num_train_data_samples);
  readDataFromFile(input_test_data_path, test_input_data_raw.get(), sizeof(MODEL_TYPE) * input_size * num_test_data_samples);

  // Read train and test targets
  readDataFromFile(input_label_train_data_path, train_target_data_raw.get(),
                   sizeof(MODEL_TYPE) * target_size * num_train_data_samples);
  readDataFromFile(input_label_test_data_path, test_target_data_raw.get(),
                   sizeof(MODEL_TYPE) * target_size * num_test_data_samples);

  // Data for train inputs and labels
  // Dim = 0 - it is number of samples
  // Dim = 1 - it is for current input

  // Train inputs
  std::vector<std::vector<MODEL_TYPE>> train_input_data(num_train_data_samples);
  for (uint32_t i = 0; i < num_train_data_samples; ++i)
  {
    train_input_data.at(i).resize(input_size);
    for (uint32_t j = 0; j < input_size; ++j)
    {
      train_input_data.at(i)[j] = *reinterpret_cast<MODEL_TYPE *>(train_input_data_raw.get() +
                                                          sizeof(MODEL_TYPE) * j +
                                                          i * sizeof(MODEL_TYPE) * input_size);
    }
  }

  // Test inputs
  std::vector<std::vector<MODEL_TYPE>> test_input_data(num_test_data_samples);
  for (uint32_t i = 0; i < num_test_data_samples; ++i)
  {
    test_input_data.at(i).resize(input_size);
    for (uint32_t j = 0; j < input_size; ++j)
    {
      test_input_data.at(i)[j] = *reinterpret_cast<MODEL_TYPE *>(test_input_data_raw.get() +
                                                             sizeof(MODEL_TYPE) * j +
                                                             i * sizeof(MODEL_TYPE) * input_size);
    }
  }

  // Train targets
  std::vector<std::vector<MODEL_TYPE>> train_target_data(num_train_data_samples);
  for (uint32_t i = 0; i < num_train_data_samples; ++i)
  {
    train_target_data.at(i).resize(target_size);
    for (uint32_t j = 0; j < target_size; ++j)
    {
      train_target_data.at(i)[j] = *reinterpret_cast<MODEL_TYPE *>(train_target_data_raw.get() +
        sizeof(MODEL_TYPE) * j +
        i * sizeof(MODEL_TYPE) * target_size);
    }
  }

  // Test targets
  std::vector<std::vector<MODEL_TYPE>> test_target_data(num_test_data_samples);
  for (uint32_t i = 0; i < num_test_data_samples; ++i)
  {
    test_target_data.at(i).resize(target_size);
    for (uint32_t j = 0; j < target_size; ++j)
    {
      test_target_data.at(i)[j] = *reinterpret_cast<MODEL_TYPE *>(test_target_data_raw.get() +
                                                              sizeof(MODEL_TYPE) * j +
                                                              i * sizeof(MODEL_TYPE) * target_size);
    }
  }

  for (uint32_t e = 0; e < training_epochs; ++e)
  {
    train_interpreter.set_training_mode(true);
    std::cout << "Run training for epoch: " << e + 1 << "/" << training_epochs << "\n";
    for (int i = 0; i < num_train_data_samples; i += config.train_config.batches)
    {
      for (int batch = 0; batch < config.train_config.batches and i + batch < num_train_data_samples; ++batch)
      {
        train_interpreter.allocateInputs();
        // Copy input data
        auto &cur_train_data = train_input_data.at(i + batch);
        auto cur_input_data = train_interpreter.getInputDataAt(0);
        std::memcpy(cur_input_data, cur_train_data.data(), sizeof(MODEL_TYPE) * input_size);

        train_interpreter.forward();

//        printPredAndTargetsValues(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
//                                  reinterpret_cast<float *>(train_target_data.at(i + batch ).data()),
//                                  target_size);
//        calculateMSE(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
//                     reinterpret_cast<float *>(train_target_data.at(i + batch).data()), target_size);
//        calculateMAE(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
//                     reinterpret_cast<float *>(train_target_data.at(i + batch).data()), target_size);

        train_interpreter.allocateTargets();

        // Copy targets values
        auto &cur_train_target_data = train_target_data.at(i + batch);
        auto cur_target_data = train_interpreter.getTargetDataAt(0);
        std::memcpy(cur_target_data, cur_train_target_data.data(),
                    sizeof(MODEL_TYPE) * target_size);

        train_interpreter.backward();
      }

      train_interpreter.updateWeights();
     // std::cout << "Update weights\n";
      //std::cout << "\n";
      train_interpreter.reset();
    }

    // Run test dataset
    std::vector<float> mse_vector;
    std::vector<float> mae_vector;

    train_interpreter.set_training_mode(false);

    std::cout << "Run test dataset for epoch: " << e + 1 << "/" << training_epochs << "\n";
    for (int i = 0; i < num_test_data_samples; i++)
    {
      train_interpreter.allocateInputs();
      // Copy input data
      auto &cur_train_data = train_input_data.at(i);
      auto cur_input_data = train_interpreter.getInputDataAt(0);
      std::memcpy(cur_input_data, cur_train_data.data(), sizeof(MODEL_TYPE) * input_size);

      train_interpreter.forward();

      printPredAndTargetsValues(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
                                reinterpret_cast<float *>(train_target_data.at(i).data()),
                                target_size);
      auto mse = calculateMSE(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
                   reinterpret_cast<float *>(train_target_data.at(i).data()), target_size);
      auto mae = calculateMAE(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)),
                              reinterpret_cast<float *>(train_target_data.at(i).data()), target_size);
      mse_vector.push_back(mse);
      mae_vector.push_back(mae);
      train_interpreter.reset();
    }

    // Calculating avg mse and mae
    float avg_mse = float(std::accumulate(mse_vector.begin(), mse_vector.end(), 0.0f)) / float(mse_vector.size());
    std::cout << "\nAverage MSE = " << avg_mse << "\n";
    float avg_mae = float(std::accumulate(mae_vector.begin(), mae_vector.end(), 0.0f)) / float(mae_vector.size());
    std::cout << "Average MAE = " << avg_mae << "\n\n";
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
