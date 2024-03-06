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

void calculateMSE(float *pred, float *target, size_t size)
{
  float mse = 0;
  for (uint32_t i = 0; i < size; ++i)
  {
    mse += (pred[i] - target[i]) * (pred[i] - target[i]);
  }
  std::cout << "Calculated MSE = " << mse / size << "\n";
}

void printDataVector(float *data, int num_samples, int num_inputs, int num_size)
{
  for (int i = 0; i < num_samples; i++)
  {
    std::cout << "Cur sample № = " << i + 1 << "\n";
    for (int j = 0; j < num_inputs; ++j)
    {
      for (int k = 0; k > num_size; ++k)
      {
        std::cout << data[k + j * num_size + i * num_size * num_inputs];
      }
    }
  }
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
  const uint32_t training_epochs = 50;
  const float lambda = 0.005f;
  const uint16_t batches = 1;

  // Configure training mode
  onert_micro::OMConfig config;
  config.wof_ptr = wof_data.data();
  config.train_mode = true;
  {
    onert_micro::OMTrainingConfig trainConfig;
    trainConfig.lambda = lambda;
    trainConfig.update_weights_in_place = batches == 1? true : false;
    trainConfig.batches = batches;

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

  // Save model inputs/outputs/target sizes
  std::vector<uint32_t> inputs_sizes(model_inputs_num);
  for (uint32_t i = 0; i < model_inputs_num; ++i)
  {
    inputs_sizes.at(i) = train_interpreter.getInputSizeAt(i);
  }

  std::vector<uint32_t> outputs_sizes(model_outputs_num);
  for (uint32_t i = 0; i < model_outputs_num; ++i)
  {
    outputs_sizes.at(i) = train_interpreter.getOutputSizeAt(i);
  }

  std::vector<uint32_t> targets_sizes(model_targets_num);
  for (uint32_t i = 0; i < model_targets_num; ++i)
  {
    targets_sizes.at(i) = train_interpreter.getTargetSizeAt(i);
  }


  char *train_input_data = new char[sizeof(MODEL_TYPE) * inputs_sizes[0] * num_train_data_samples];
  char *train_target_data = new char[sizeof(MODEL_TYPE) * targets_sizes[0] * num_train_data_samples];

  char *test_input_data = new char[sizeof(MODEL_TYPE) * inputs_sizes[0] * num_test_data_samples];
  char *test_target_data = new char[sizeof(MODEL_TYPE) * targets_sizes[0] * num_test_data_samples];

  readDataFromFile(input_train_data_path, train_input_data, sizeof(MODEL_TYPE) * inputs_sizes[0] * num_train_data_samples);
  //readDataFromFile(input_test_data_path, test_input_data, sizeof(MODEL_TYPE) * inputs_sizes[0] * num_test_data_samples);

  readDataFromFile(input_label_train_data_path, train_target_data,
                   sizeof(MODEL_TYPE) * targets_sizes[0] * num_train_data_samples);
//  readDataFromFile(input_label_test_data_path, test_target_data,
//                   sizeof(MODEL_TYPE) * targets_sizes[0] * num_test_data_samples);

  // Data for train inputs and labels
  // Dim = 0 - it is number of samples
  // Dim = 1 - it is number of inputs
  // Dim = 2 - it is for current input

  // Train inputs
  std::vector<std::vector<std::vector<char>>> trains_input_data(num_train_data_samples);
  for (uint32_t i = 0; i < num_train_data_samples; ++i)
  {
    trains_input_data.at(i).resize(model_inputs_num);
    for (uint32_t j = 0; j < model_inputs_num; ++j)
    {
      auto size = sizeof(MODEL_TYPE) * inputs_sizes[j];
      trains_input_data.at(i).at(j).resize(sizeof(MODEL_TYPE) * inputs_sizes[j]);
      std::memcpy(trains_input_data.at(i).at(j).data(), train_input_data + size * j + size * model_inputs_num * i, size);
    }
  }

  // Train labels
  std::vector<std::vector<std::vector<char>>> train_labels_data(num_train_data_samples);
  for (uint32_t i = 0; i < num_train_data_samples; ++i)
  {
    train_labels_data.at(i).resize(model_targets_num);
    for (uint32_t j = 0; j < model_targets_num; ++j)
    {
      auto size = sizeof(MODEL_TYPE) * targets_sizes[j];
      train_labels_data.at(i).at(j).resize(sizeof(MODEL_TYPE) * targets_sizes[j]);
      std::memcpy(train_labels_data.at(i).at(j).data(), train_target_data + size * j + size * model_targets_num * i, size);
    }
  }

//  // Test inputs
//  std::vector<std::vector<std::vector<char>>> test_inputs_data(num_test_data_samples);
//  for (uint32_t i = 0; i < num_test_data_samples; ++i)
//  {
//    test_inputs_data.at(i).reserve(model_inputs_num);
//    for (uint32_t j = 0; j < model_inputs_num; ++j)
//    {
//      auto size = sizeof(MODEL_TYPE) * inputs_sizes[j];
//      test_inputs_data.at(i).at(j).reserve(sizeof(MODEL_TYPE) * inputs_sizes[j]);
//      std::memcpy(test_inputs_data.at(i).at(j).data(), test_input_data + size * j + size * model_inputs_num * i, size);
//    }
//  }
//
//  // Test labels
//  std::vector<std::vector<std::vector<char>>> test_labels_data(num_test_data_samples);
//  for (uint32_t i = 0; i < num_test_data_samples; ++i)
//  {
//    test_labels_data.at(i).reserve(model_targets_num);
//    for (uint32_t j = 0; j < model_targets_num; ++j)
//    {
//      auto size = sizeof(MODEL_TYPE) * targets_sizes[j];
//      test_labels_data.at(i).at(j).reserve(sizeof(MODEL_TYPE) * targets_sizes[j]);
//      std::memcpy(test_labels_data.at(i).at(j).data(), test_target_data + size * j + size * model_targets_num * i, size);
//    }
//  }

  printf("Run train dataset:\n");
  for (int i = 0; i < num_train_data_samples; i+=batches)
  {
    train_interpreter.reset();
    assert(trains_input_data.at(i).size() == 1);
    for (uint32_t b = 0; b < batches; ++b)
    {
      train_interpreter.allocateInputs();
      for (uint32_t j = 0; j < model_inputs_num; ++j)
      {
        // Copy input data
        auto &cur_train_data = trains_input_data.at(i + b).at(j);
        auto cur_input_data = train_interpreter.getInputDataAt(j);
        std::memcpy(cur_input_data, cur_train_data.data(), sizeof(MODEL_TYPE) * inputs_sizes[j]);
      }
      train_interpreter.forward();
      printPredAndTargetsValues(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)), reinterpret_cast<float*>(train_labels_data.at(i + b).at(0).data()), targets_sizes[0]);
      calculateMSE(reinterpret_cast<float *>(train_interpreter.getOutputDataAt(0)), reinterpret_cast<float*>(train_labels_data.at(i + b).at(0).data()), targets_sizes[0]);
      train_interpreter.allocateTargets();
      for (uint32_t j = 0; j < model_targets_num; ++j)
      {
        auto &cur_train_target_data = train_labels_data.at(i + b).at(j);
        auto cur_target_data = train_interpreter.getTargetDataAt(j);
        std::memcpy(cur_target_data, cur_train_target_data.data(), sizeof(MODEL_TYPE) * targets_sizes[j]);
      }
      train_interpreter.backward();
    }
    train_interpreter.updateWeights();

  }

//  auto test_data_u8 = reinterpret_cast<char *>(test_data);
//
//  printf("Run test dataset:\n");
//  for (int i = 0; i < num_test_data_samples; ++i)
//  {
//    auto input_data = reinterpret_cast<char *>(interpreter.allocateInputTensor(0));
//
//
//    std::memcpy(input_data, test_data_u8, size);
//
//    interpreter.interpret();
//    auto data = reinterpret_cast<float *>(interpreter.readOutputTensor(0));
//
//    printf("Sample № = %d\n", i);
//    for (int j = 0; j < output_size / sizeof(float); ++j)
//    {
//      printf("j = %d: predicted_result = %f, correct_result = %f\n", j, data[j],
//             reinterpret_cast<float *>(label_test_data)[j + i * output_size / sizeof(float)]);
//    }
//    printf("\n");
//    test_data_u8 += size;
//  }
//
//  float mse_result = 0.0f;
//
//  settings.metric = luci_interpreter::training::MSE;
//  onert_micro_training.test(num_train_data_samples, reinterpret_cast<const uint8_t *>(train_data),
//                            reinterpret_cast<const uint8_t *>(label_train_data),
//                            reinterpret_cast<void *>(&mse_result));
//
//  float mae_result = 0.0f;
//
//  settings.metric = luci_interpreter::training::MAE;
//  onert_micro_training.test(num_train_data_samples, reinterpret_cast<const uint8_t *>(train_data),
//                            reinterpret_cast<const uint8_t *>(label_train_data),
//                            reinterpret_cast<void *>(&mae_result));
//
//  printf("MSE_ERROR TRAIN = %f\n", mse_result);
//
//  printf("MAE_ERROR TRAIN = %f\n", mae_result);
//
//  mse_result = 0.0f;
//
//  settings.metric = luci_interpreter::training::MSE;
//  onert_micro_training.test(num_test_data_samples, reinterpret_cast<const uint8_t *>(test_data),
//                            reinterpret_cast<const uint8_t *>(label_test_data),
//                            reinterpret_cast<void *>(&mse_result));
//
//  mae_result = 0.0f;
//
//  settings.metric = luci_interpreter::training::MAE;
//  onert_micro_training.test(num_test_data_samples, reinterpret_cast<const uint8_t *>(test_data),
//                            reinterpret_cast<const uint8_t *>(label_test_data),
//                            reinterpret_cast<void *>(&mae_result));
//
//  printf("MSE_ERROR TEST = %f\n", mse_result);
//
//  printf("MAE_ERROR TEST = %f\n", mae_result);

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
