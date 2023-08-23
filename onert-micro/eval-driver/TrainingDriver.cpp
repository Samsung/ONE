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

#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/TrainingOnertMicro.h>
#include <luci_interpreter/TrainingSettings.h>

#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

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

} // namespace

/*
 * @brief EvalDriver main
 *
 *        Driver for testing luci-inerpreter
 *
 */
int entry(int argc, char **argv)
{
  if (argc != 8)
  {
    std::cerr
      << "Usage: " << argv[0]
      << " <path/to/circle/model> <path/to/input/train_data> <path/to/input/label_train_data> "
         "<path/to/input/test_data> <path/to/input/label_test_data> num_of_train_smpl "
         "num_of_test_smpl\n";
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
  const char *input_train_data_path = argv[2];
  const char *input_label_train_data_path = argv[3];
  const char *input_test_data_path = argv[4];
  const char *input_label_test_data_path = argv[5];
  const int32_t num_train_data_samples = atoi(argv[6]);
  const int32_t num_test_data_samples = atoi(argv[7]);

  std::ifstream file(filename, std::ios::binary | std::ios::in);
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

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(model_data.data(), true);

  luci_interpreter::training::TrainingSettings settings;
  settings.learning_rate = 0.0001;
  settings.number_of_epochs = 100;
  settings.batch_size = 1;

  const auto input_size = interpreter.getInputDataSizeByIndex(0);
  const auto output_size = interpreter.getOutputDataSizeByIndex(0);

  char *train_data = new char[input_size * num_train_data_samples];
  char *label_train_data = new char[output_size * num_train_data_samples];

  char *test_data = new char[input_size * num_test_data_samples];
  char *label_test_data = new char[output_size * num_test_data_samples];

  readDataFromFile(input_train_data_path, train_data, input_size * num_train_data_samples);
  readDataFromFile(input_test_data_path, test_data, input_size * num_test_data_samples);

  readDataFromFile(input_label_train_data_path, label_train_data,
                   output_size * num_train_data_samples);
  readDataFromFile(input_label_test_data_path, label_test_data,
                   output_size * num_test_data_samples);

  luci_interpreter::training::TrainingOnertMicro onert_micro_training(&interpreter, settings);
  onert_micro_training.enableTrainingMode();
  onert_micro_training.train(num_train_data_samples, reinterpret_cast<uint8_t *>(train_data),
                             reinterpret_cast<uint8_t *>(label_train_data));
  onert_micro_training.disableTrainingMode();

  auto size = interpreter.getInputDataSizeByIndex(0);
  auto train_data_u8 = reinterpret_cast<char *>(train_data);

  printf("Run train dataset:\n");
  for (int i = 0; i < num_train_data_samples; ++i)
  {
    auto input_data = reinterpret_cast<char *>(interpreter.allocateInputTensor(0));

    std::memcpy(input_data, train_data_u8, size);

    interpreter.interpret();
    auto data = reinterpret_cast<float *>(interpreter.readOutputTensor(0));

    printf("Sample № = %d\n", i);
    for (int j = 0; j < output_size / sizeof(float); ++j)
    {
      printf("j = %d: predicted_result = %f, correct_result = %f\n", j, data[j],
             reinterpret_cast<float *>(label_train_data)[j + i * output_size / sizeof(float)]);
    }
    printf("\n");
    train_data_u8 += size;
  }

  auto test_data_u8 = reinterpret_cast<char *>(test_data);

  printf("Run test dataset:\n");
  for (int i = 0; i < num_test_data_samples; ++i)
  {
    auto input_data = reinterpret_cast<char *>(interpreter.allocateInputTensor(0));

    std::memcpy(input_data, test_data_u8, size);

    interpreter.interpret();
    auto data = reinterpret_cast<float *>(interpreter.readOutputTensor(0));

    printf("Sample № = %d\n", i);
    for (int j = 0; j < output_size / sizeof(float); ++j)
    {
      printf("j = %d: predicted_result = %f, correct_result = %f\n", j, data[j],
             reinterpret_cast<float *>(label_test_data)[j + i * output_size / sizeof(float)]);
    }
    printf("\n");
    test_data_u8 += size;
  }

  float mse_result = 0.0f;

  settings.metric = luci_interpreter::training::MSE;
  onert_micro_training.test(num_train_data_samples, reinterpret_cast<const uint8_t *>(train_data),
                            reinterpret_cast<const uint8_t *>(label_train_data),
                            reinterpret_cast<void *>(&mse_result));

  float mae_result = 0.0f;

  settings.metric = luci_interpreter::training::MAE;
  onert_micro_training.test(num_train_data_samples, reinterpret_cast<const uint8_t *>(train_data),
                            reinterpret_cast<const uint8_t *>(label_train_data),
                            reinterpret_cast<void *>(&mae_result));

  printf("MSE_ERROR TRAIN = %f\n", mse_result);

  printf("MAE_ERROR TRAIN = %f\n", mae_result);

  mse_result = 0.0f;

  settings.metric = luci_interpreter::training::MSE;
  onert_micro_training.test(num_test_data_samples, reinterpret_cast<const uint8_t *>(test_data),
                            reinterpret_cast<const uint8_t *>(label_test_data),
                            reinterpret_cast<void *>(&mse_result));

  mae_result = 0.0f;

  settings.metric = luci_interpreter::training::MAE;
  onert_micro_training.test(num_test_data_samples, reinterpret_cast<const uint8_t *>(test_data),
                            reinterpret_cast<const uint8_t *>(label_test_data),
                            reinterpret_cast<void *>(&mae_result));

  printf("MSE_ERROR TEST = %f\n", mse_result);

  printf("MAE_ERROR TEST = %f\n", mae_result);

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
