/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdint>
#include <signal.h>

#include <tensorflow/lite/context.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>

#include "labels.h"
#include "tflite/ext/nnapi_delegate.h"
#include "tflite/ext/kernels/register.h"

const std::string kDefaultImagesDir = "res/input/";
const std::string kDefaultModelFile = "res/model.tflite";

template <typename... Args> void Print(const char *fmt, Args... args)
{
#if __cplusplus >= 201703L
  std::cerr << boost::str(boost::format(fmt) % ... % std::forward<Args>(args)) << std::endl;
#else
  boost::format f(fmt);
  using unroll = int[];
  unroll{0, (f % std::forward<Args>(args), 0)...};
  std::cerr << boost::str(f) << std::endl;
#endif
}

template <typename DataType> struct BaseLabelData
{
  explicit BaseLabelData(int label = -1, DataType confidence = 0)
    : label(label), confidence(confidence)
  {
  }

  static std::vector<BaseLabelData<DataType>> FindLabels(const DataType *output_tensor,
                                                         unsigned int top_n = 5)
  {
    top_n = top_n > 1000 ? 1000 : top_n;
    size_t n = 0;
    std::vector<size_t> indices(1000);
    std::generate(indices.begin(), indices.end(), [&n]() { return n++; });
    std::sort(indices.begin(), indices.end(), [output_tensor](const size_t &i1, const size_t &i2) {
      return output_tensor[i1] > output_tensor[i2];
    });
    std::vector<BaseLabelData<DataType>> results(top_n);
    for (unsigned int i = 0; i < top_n; ++i)
    {
      results[i].label = indices[i];
      results[i].confidence = output_tensor[indices[i]];
    }
    return results;
  }

  int label;
  DataType confidence;
};

class BaseRunner
{
public:
  virtual ~BaseRunner() = default;

  /**
   * @brief Run a model for each file in a directory, and collect and print
   * statistics.
   */
  virtual void IterateInDirectory(const std::string &dir_path, const int labels_offset) = 0;

  /**
   * @brief Request that the iteration be stopped after the current file.
   */
  virtual void ScheduleInterruption() = 0;
};

template <typename DataType_> class Runner : public BaseRunner
{
public:
  using DataType = DataType_;
  using LabelData = BaseLabelData<DataType>;

  const int kInputSize;
  const int KOutputSize = 1001 * sizeof(DataType);

  Runner(std::unique_ptr<tflite::Interpreter> interpreter,
         std::unique_ptr<tflite::FlatBufferModel> model,
         std::unique_ptr<::nnfw::tflite::NNAPIDelegate> delegate, unsigned img_size)
    : interpreter(std::move(interpreter)), model(std::move(model)), delegate(std::move(delegate)),
      interrupted(false), kInputSize(1 * img_size * img_size * 3 * sizeof(DataType))
  {
    inference_times.reserve(500);
    top1.reserve(500);
    top5.reserve(500);
  }

  virtual ~Runner() = default;

  /**
   * @brief Get the model's input tensor.
   */
  virtual DataType *GetInputTensor() = 0;

  /**
   * @brief Get the model's output tensor.
   */
  virtual DataType *GetOutputTensor() = 0;

  /**
   * @brief Load Image file into tensor.
   * @return Class number if present in filename, -1 otherwise.
   */
  virtual int LoadFile(const boost::filesystem::path &input_file)
  {
    DataType *input_tensor = GetInputTensor();
    if (input_file.extension() == ".bin")
    {
      // Load data as raw tensor
      std::ifstream input_stream(input_file.string(), std::ifstream::binary);
      input_stream.read(reinterpret_cast<char *>(input_tensor), kInputSize);
      input_stream.close();
      int class_num = boost::lexical_cast<int>(input_file.filename().string().substr(0, 4));
      return class_num;
    }
    else
    {
      // Load data as image file
      throw std::runtime_error("Runner can only load *.bin files");
    }
  }

  void Invoke()
  {
    TfLiteStatus status;
    if (delegate)
    {
      status = delegate->Invoke(interpreter.get());
    }
    else
    {
      status = interpreter->Invoke();
    }
    if (status != kTfLiteOk)
    {
      throw std::runtime_error("Failed to invoke interpreter.");
    }
  }

  int Process()
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    Invoke();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fs = t1 - t0;
    auto d = std::chrono::duration_cast<std::chrono::milliseconds>(fs);
    inference_times.push_back(d.count());
    if (d > std::chrono::milliseconds(10))
    {
      Print("  -- inference duration: %lld ms", d.count());
    }
    else
    {
      auto du = std::chrono::duration_cast<std::chrono::microseconds>(fs);
      Print("  -- inference duration: %lld us", du.count());
    }
    return 0;
  }

  void DumpOutputTensor(const std::string &output_file)
  {
    DataType *output_tensor = GetOutputTensor();
    std::ofstream output_stream(output_file, std::ofstream::binary);
    output_stream.write(reinterpret_cast<char *>(output_tensor), KOutputSize);
  }

  void PrintExecutionSummary() const
  {
    Print("Execution summary:");
    Print("  -- # of processed images: %d", num_images);
    if (num_images == 0)
    {
      return;
    }
    // Inference time - mean
    double mean = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / num_images;
    Print("  -- mean inference time: %.1f ms", mean);
    // Inference time - std
    std::vector<double> diff(num_images);
    std::transform(inference_times.begin(), inference_times.end(), diff.begin(),
                   [mean](size_t n) { return n - mean; });
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double std_inference_time = std::sqrt(sq_sum / num_images);
    Print("  -- std inference time: %.1f ms", std_inference_time);
    // Top-1 and Top-5 accuracies
    float num_top1 = std::accumulate(top1.begin(), top1.end(), 0);
    float num_top5 = std::accumulate(top5.begin(), top5.end(), 0);
    Print("  -- top1: %.3f, top5: %.3f", num_top1 / num_images, num_top5 / num_images);
  }

  virtual void ScheduleInterruption() override { interrupted = true; }

  virtual void IterateInDirectory(const std::string &dir_path, const int labels_offset) override
  {
    interrupted = false;
    namespace fs = boost::filesystem;
    if (!fs::is_directory(dir_path))
    {
      throw std::runtime_error("Could not open input directory.");
    }

    inference_times.clear();
    top1.clear();
    top5.clear();
    int class_num;
    num_images = 0;
    std::vector<LabelData> lds;
    fs::directory_iterator end;
    for (auto it = fs::directory_iterator(dir_path); it != end; ++it)
    {
      if (interrupted)
      {
        break;
      }
      if (!fs::is_regular_file(*it))
      {
        continue;
      }
      Print("File : %s", it->path().string());
      try
      {
        class_num = LoadFile(*it) + labels_offset;
        Print("Class: %d", class_num);
      }
      catch (std::exception &e)
      {
        Print("%s", e.what());
        continue;
      }
      int status = Process();
      if (status == 0)
      {
        DataType *output_tensor = GetOutputTensor();
        lds = LabelData::FindLabels(output_tensor, 5);
        bool is_top1 = lds[0].label == class_num;
        bool is_top5 = false;
        for (const auto &ld : lds)
        {
          is_top5 = is_top5 || (ld.label == class_num);
          Print("  -- label: %s (%d), prob: %.3f", ld.label >= 0 ? labels[ld.label] : "", ld.label,
                static_cast<float>(ld.confidence));
        }
        Print("  -- top1: %d, top5: %d", is_top1, is_top5);
        top1.push_back(is_top1);
        top5.push_back(is_top5);
      }
      ++num_images;
    }
    PrintExecutionSummary();
  }

protected:
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<::nnfw::tflite::NNAPIDelegate> delegate;

  std::vector<size_t> inference_times;
  std::vector<bool> top1;
  std::vector<bool> top5;
  uint num_images;
  std::atomic_bool interrupted;
};

class FloatRunner : public Runner<float>
{
public:
  using Runner<float>::DataType;

  FloatRunner(std::unique_ptr<tflite::Interpreter> interpreter,
              std::unique_ptr<tflite::FlatBufferModel> model,
              std::unique_ptr<::nnfw::tflite::NNAPIDelegate> delegate, unsigned img_size)
    : Runner<float>(std::move(interpreter), std::move(model), std::move(delegate), img_size)
  {
  }

  virtual ~FloatRunner() = default;

  virtual DataType *GetInputTensor() override
  {
    return interpreter->tensor(interpreter->inputs()[0])->data.f;
  }

  virtual DataType *GetOutputTensor() override
  {
    return interpreter->tensor(interpreter->outputs()[0])->data.f;
  }
};

class QuantizedRunner : public Runner<uint8_t>
{
public:
  using Runner<uint8_t>::DataType;

  QuantizedRunner(std::unique_ptr<tflite::Interpreter> interpreter,
                  std::unique_ptr<tflite::FlatBufferModel> model,
                  std::unique_ptr<::nnfw::tflite::NNAPIDelegate> delegate, unsigned img_size)
    : Runner<uint8_t>(std::move(interpreter), std::move(model), std::move(delegate), img_size)
  {
  }

  virtual ~QuantizedRunner() = default;

  virtual DataType *GetInputTensor() override
  {
    return interpreter->tensor(interpreter->inputs()[0])->data.uint8;
  }

  virtual DataType *GetOutputTensor() override
  {
    return interpreter->tensor(interpreter->outputs()[0])->data.uint8;
  }
};

enum class Target
{
  TfLiteCpu,      /**< Use Tensorflow Lite's CPU kernels. */
  TfLiteDelegate, /**< Use Tensorflow Lite's NN API delegate. */
  NnfwDelegate    /**< Use NNFW's NN API delegate. */
};

std::unique_ptr<BaseRunner> MakeRunner(const std::string &model_path, unsigned img_size,
                                       Target target = Target::NnfwDelegate)
{
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (not model)
  {
    throw std::runtime_error(model_path + ": file not found or corrupted.");
  }
  Print("Model loaded.");

  std::unique_ptr<tflite::Interpreter> interpreter;
  nnfw::tflite::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (not interpreter)
  {
    throw std::runtime_error("interpreter construction failed.");
  }
  if (target == Target::TfLiteCpu)
  {
    interpreter->SetNumThreads(std::max(std::thread::hardware_concurrency(), 1U));
  }
  else
  {
    interpreter->SetNumThreads(1);
  }
  if (target == Target::TfLiteDelegate)
  {
    interpreter->UseNNAPI(true);
  }

  int input_index = interpreter->inputs()[0];
  interpreter->ResizeInputTensor(input_index,
                                 {1, static_cast<int>(img_size), static_cast<int>(img_size), 3});
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    throw std::runtime_error("tensor allocation failed.");
  }

  if (target == Target::TfLiteDelegate)
  {
    // Do a fake run to load NN API functions.
    interpreter->Invoke();
  }

  std::unique_ptr<::nnfw::tflite::NNAPIDelegate> delegate;
  if (target == Target::NnfwDelegate)
  {
    delegate.reset(new ::nnfw::tflite::NNAPIDelegate);
    delegate->BuildGraph(&(interpreter.get()->primary_subgraph()));
  }

  if (interpreter->tensor(input_index)->type == kTfLiteFloat32)
  {
    return std::unique_ptr<FloatRunner>(
      new FloatRunner(std::move(interpreter), std::move(model), std::move(delegate), img_size));
  }
  else if (interpreter->tensor(input_index)->type == kTfLiteUInt8)
  {
    return std::unique_ptr<QuantizedRunner>(
      new QuantizedRunner(std::move(interpreter), std::move(model), std::move(delegate), img_size));
  }
  throw std::invalid_argument("data type of model's input tensor is not supported.");
}

Target GetTarget(const std::string &str)
{
  static const std::map<std::string, Target> target_names{
    {"tflite-cpu", Target::TfLiteCpu},
    {"tflite-delegate", Target::TfLiteDelegate},
    {"nnfw-delegate", Target::NnfwDelegate}};
  if (target_names.find(str) == target_names.end())
  {
    throw std::invalid_argument(
      str + ": invalid target. Run with --help for a list of available targets.");
  }
  return target_names.at(str);
}

// We need a global pointer to the runner for the SIGINT handler
BaseRunner *runner_ptr = nullptr;
void HandleSigInt(int)
{
  if (runner_ptr != nullptr)
  {
    Print("Interrupted. Execution will stop after current image.");
    runner_ptr->ScheduleInterruption();
    runner_ptr = nullptr;
  }
  else
  {
    exit(1);
  }
}

int main(int argc, char *argv[])
try
{
  namespace po = boost::program_options;
  po::options_description desc("Run a model on multiple binary images and print"
                               " statistics");
  desc.add_options()("help", "print this message and quit")(
    "model", po::value<std::string>()->default_value(kDefaultModelFile), "tflite file")(
    "input", po::value<std::string>()->default_value(kDefaultImagesDir),
    "directory with input images")("offset", po::value<int>()->default_value(1), "labels offset")(
    "target", po::value<std::string>()->default_value("nnfw-delegate"),
    "how the model will be run (available targets: tflite-cpu, "
    "tflite-delegate, nnfw-delegate)")("imgsize", po::value<unsigned>()->default_value(224),
                                       "the width and height of the image");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help"))
  {
    std::cerr << desc << std::endl;
    return 0;
  }

  auto runner = MakeRunner(vm["model"].as<std::string>(), vm["imgsize"].as<unsigned>(),
                           GetTarget(vm["target"].as<std::string>()));
  runner_ptr = runner.get();

  struct sigaction sigint_handler;
  sigint_handler.sa_handler = HandleSigInt;
  sigemptyset(&sigint_handler.sa_mask);
  sigint_handler.sa_flags = 0;
  sigaction(SIGINT, &sigint_handler, nullptr);

  Print("Running TensorFlow Lite...");
  runner->IterateInDirectory(vm["input"].as<std::string>(), vm["offset"].as<int>());
  Print("Done.");
  return 0;
}
catch (std::exception &e)
{
  Print("%s: %s", argv[0], e.what());
  return 1;
}
