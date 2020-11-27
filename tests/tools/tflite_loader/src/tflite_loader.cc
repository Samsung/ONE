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

#include "tflite/ext/kernels/register.h"

#include "args.h"
#include "tflite/InterpreterSession.h"
#include "tflite/Assert.h"
#include "tflite/Diff.h"
#include "misc/tensor/IndexIterator.h"

#include <iostream>
#include <fstream>

#include "compiler/Compiler.h"
#include "exec/Execution.h"
#include "ir/Graph.h"

#include "tflite_loader.h"

#include <memory>

const int RUN_FAILED = 1;

using namespace tflite;
using namespace nnfw::tflite;

const int FILE_ERROR = 2;
const float DIFFERENCE_THRESHOLD = 10e-5;

// Read vector of floats from selected file
std::vector<float> readData(const string &path)
{
  std::ifstream in(path);
  if (!in.good())
  {
    std::cerr << "can not open data file " << path << "\n";
    exit(FILE_ERROR);
  }
  in.seekg(0, std::ifstream::end);
  size_t len = in.tellg();
  in.seekg(0, std::ifstream::beg);
  assert(len % sizeof(float) == 0);
  size_t size = len / sizeof(float);
  std::vector<float> vec(size);
  for (size_t i = 0; i < size; ++i)
  {
    in.read(reinterpret_cast<char *>(&vec[i]), sizeof(float));
  }
  return vec;
}

std::vector<float> randomData(nnfw::misc::RandomGenerator &randgen, const uint64_t size)
{
  std::vector<float> vec(size);
  for (uint64_t i = 0; i < size; i++)
  {
    vec[i] = randgen.generate<float>();
  }
  return vec;
}

void executeGraph(const std::shared_ptr<onert::ir::Graph> &g,
                  const std::vector<std::vector<float>> &inputs,
                  std::vector<std::vector<float>> &outputs)
{
  auto subgs = std::make_shared<onert::ir::Subgraphs>();
  subgs->push(onert::ir::SubgraphIndex{0}, g);
  auto compiler = new onert::compiler::Compiler(subgs);
  std::shared_ptr<onert::exec::ExecutorMap> executors;
  // Compilation
  try
  {
    executors = compiler->compile();
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Execution] Can't compile model" << std::endl;
    std::cerr << e.what() << std::endl;
    exit(-1);
  }

  std::cout << "[Execution] Graph compiled!" << std::endl;

  auto execution = std::make_shared<onert::exec::Execution>(executors);

  // Setting IO
  try
  {
    // Verify input shapes
    auto num_inputs = inputs.size();
    for (size_t i = 0; i < num_inputs; i++)
    {
      auto input_operand_idx = g->getInputs().at(i);
      auto input_shape = g->operands().at(input_operand_idx).shape();
      assert(inputs[i].size() == input_shape.num_elements());
    }

    // Set output shapes
    auto num_outputs = g->getOutputs().size();
    outputs.resize(num_outputs);
    for (uint32_t i = 0; i < num_outputs; i++)
    {
      auto output_operand_idx = g->getOutputs().at(i);
      auto output_shape = g->operands().at(output_operand_idx).shape();
      outputs[i].resize(output_shape.num_elements());
    }

    for (size_t i = 0; i < num_inputs; i++)
      execution->setInput(onert::ir::IOIndex(i), inputs[i].data(),
                          inputs[i].size() * sizeof(float));
    for (uint32_t i = 0; i < num_outputs; i++)
      execution->setOutput(onert::ir::IOIndex(i), outputs[i].data(),
                           outputs[i].size() * sizeof(float));
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Execution] Can't set model IO" << std::endl;
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  try
  {
    execution->execute();
  }
  catch (const std::exception &e)
  {
    std::cerr << "[Execution] Can't execute" << std::endl;
    std::cerr << e.what() << '\n';
    exit(-1);
  }

  std::cout << "[Execution] Done!" << std::endl;

  delete compiler;
}

int main(const int argc, char **argv)
{
  TFLiteRun::Args args(argc, argv);

  auto tflite_file = args.getTFLiteFilename();
  auto data_files = args.getDataFilenames();

  if (tflite_file.empty())
  {
    args.print(argv);
    return RUN_FAILED;
  }

  std::cout << "[Execution] Stage start!" << std::endl;
  std::shared_ptr<onert::ir::Graph> test_graph;
  // Loading
  try
  {
    test_graph =
      onert::tflite_loader::loadModel(tflite_file.c_str())->at(onert::ir::SubgraphIndex{0});
  }
  catch (std::exception &e)
  {
    std::cerr << "[ ERROR ] "
              << "Failure during model load" << std::endl;
    std::cerr << e.what() << std::endl;
    exit(-1);
  }

  // TODO: Support another input/output types
  for (const auto &input_idx : test_graph->getInputs())
  {
    const auto input_type = test_graph->operands().at(input_idx).typeInfo().type();
    assert(input_type == onert::ir::DataType::FLOAT32 && "Only FLOAT32 inputs are supported");
  }
  for (const auto &output_idx : test_graph->getOutputs())
  {
    const auto output_type = test_graph->operands().at(output_idx).typeInfo().type();
    assert(output_type == onert::ir::DataType::FLOAT32 && "Only FLOAT32 outputs are supported");
  }

  std::cout << "[Execution] Model is deserialized!" << std::endl;
  auto num_inputs = test_graph->getInputs().size();
  std::vector<std::vector<float>> inputs(num_inputs);
  bool generate_data = data_files.empty();
  bool read_data = data_files.size() == num_inputs;
  if (num_inputs == 0)
  {
    std::cerr << "[ ERROR ] "
              << "No inputs in model => execution is not possible" << std::endl;
    exit(1);
  }
  if (!generate_data && !read_data)
  {
    std::cerr << "[ ERROR ] "
              << "Wrong number of input files." << std::endl;
    exit(1);
  }

  const int seed = 1; /* TODO Add an option for seed value */
  nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};
  try
  {
    for (uint32_t i = 0; i < num_inputs; i++)
    {
      if (generate_data)
      {
        uint64_t sz =
          test_graph->operands().at(test_graph->getInputs().at(i)).shape().num_elements();
        inputs[i] = randomData(randgen, sz);
      }
      else /* read_data */
        inputs[i] = readData(data_files[i]);
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "[ ERROR ] "
              << "Failure during input data generation" << std::endl;
    std::cerr << e.what() << std::endl;
    exit(-1);
  }

  std::cout << "[Execution] Input data is defined!" << std::endl;
  std::vector<std::vector<float>> outputs;
  // Run graph
  executeGraph(test_graph, inputs, outputs);
  // Compare with tflite
  std::cout << "[Comparison] Stage start!" << std::endl;
  // Read tflite model
  StderrReporter error_reporter;
  auto model = FlatBufferModel::BuildFromFile(tflite_file.c_str(), &error_reporter);

  BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;
  try
  {
    TFLITE_ENSURE(builder(&interpreter));
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    exit(FILE_ERROR);
  }
  interpreter->SetNumThreads(2);

  auto sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter.get());
  sess->prepare();
  // Set input and run
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto input_tensor = interpreter->tensor(interpreter->inputs().at(i));
    memcpy(input_tensor->data.f, inputs[i].data(), inputs[i].size() * sizeof(float));
  }
  if (!sess->run())
  {
    std::cout << "[Comparison] TFLite run failed!" << std::endl;
    assert(0 && "Run failed!");
  }
  std::cout << "[Comparison] TFLite run done!" << std::endl;

  // Calculate max difference over all outputs
  float max_difference = 0.0f;
  auto num_outputs = test_graph->getOutputs().size();
  for (uint32_t out_idx = 0; out_idx < num_outputs; out_idx++)
  {
    const auto &tflite_output_tensor = interpreter->tensor(interpreter->outputs().at(out_idx));
    const auto &nnfw_output_tensor = outputs[out_idx];

    if (nnfw_output_tensor.size() != tflite_output_tensor->bytes / sizeof(float))
      std::cout << "[Comparison] Different size of outputs!" << std::endl;
    // Check max difference
    float *tflite_out_ptr = tflite_output_tensor->data.f;
    for (const auto &nnfw_out : nnfw_output_tensor)
    {
      if (std::abs(nnfw_out - *tflite_out_ptr) > max_difference)
        max_difference = std::abs(nnfw_out - *tflite_out_ptr);

      tflite_out_ptr++;
    }
  }

  // Print results
  std::cout << "[Comparison] Max difference: " << max_difference << std::endl;
  int ret = 0;
  if (max_difference > DIFFERENCE_THRESHOLD)
  {
    std::cout << "[Comparison] Outputs is not equal!" << std::endl;
    ret = 1;
  }
  else
  {
    std::cout << "[Comparison] Outputs is equal!" << std::endl;
  }
  std::cout << "[Comparison] Done!" << std::endl;

  return ret;
}
