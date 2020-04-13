/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <tensorflow/lite/model.h>
#include <tensorflow/core/public/session.h>

#include "tflite/Assert.h"
#include "tflite/Session.h"
#include "tflite/InterpreterSession.h"
#include "tflite/NNAPISession.h"
#include "tflite/ext/kernels/register.h"

#include "misc/fp32.h"

#include <iostream>

#include <string>
#include <vector>

#define TF_ENSURE(e)                               \
  {                                                \
    if (!(e).ok())                                 \
    {                                              \
      throw std::runtime_error{"'" #e "' FAILED"}; \
    }                                              \
  }

using namespace tflite;
using namespace tflite::ops::builtin;

std::unique_ptr<FlatBufferModel> BuildModelFromFile(const std::string &path)
{
  static StderrReporter reporter;
  return FlatBufferModel::BuildFromFile(path.c_str(), &reporter);
}

std::unique_ptr<Interpreter> BuildInterpFromModel(const std::unique_ptr<FlatBufferModel> &model)
{
  std::unique_ptr<Interpreter> interp;

  BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);

  TFLITE_ENSURE(builder(&interp));

  return std::move(interp);
}

tensorflow::TensorShape asTensorflowShape(const TfLiteTensor *tensor)
{
  tensorflow::TensorShape shape;

  const int rank = tensor->dims->size;

  for (int axis = 0; axis < rank; ++axis)
  {
    shape.AddDim(tensor->dims->data[axis]);
  }

  return shape;
}

uint32_t count_elements(const TfLiteTensor *tensor)
{
  const int rank = tensor->dims->size;

  if (rank == 0)
  {
    return 0;
  }

  uint32_t res = 1;

  for (int axis = 0; axis < rank; ++axis)
  {
    res *= tensor->dims->data[axis];
  }

  return res;
}

int main(int argc, char **argv)
{
  const bool use_nnapi = nnfw::misc::EnvVar("USE_NNAPI").asBool(false);

  if (argc < 3)
  {
    std::cerr << "USAGE: " << argv[0] << " [T/F lite model] [T/F model]" << std::endl;
    return 255;
  }

  //
  // Prepare Tensorflow Lite session
  //
  const std::string lite_model_path{argv[1]};

  auto lite_model = BuildModelFromFile(lite_model_path);
  auto lite_interp = BuildInterpFromModel(lite_model);

  std::shared_ptr<nnfw::tflite::Session> lite_sess;

  if (use_nnapi)
  {
    lite_sess = std::make_shared<nnfw::tflite::NNAPISession>(lite_interp.get());
  }
  else
  {
    lite_sess = std::make_shared<nnfw::tflite::InterpreterSession>(lite_interp.get());
  }

  //
  // Prepare Tensorflow session
  //
  const std::string full_model_path{argv[2]};

  tensorflow::Session *full_sess;
  tensorflow::GraphDef full_model;

  TF_ENSURE(tensorflow::NewSession(tensorflow::SessionOptions(), &full_sess));
  TF_ENSURE(ReadBinaryProto(tensorflow::Env::Default(), full_model_path, &full_model));
  TF_ENSURE(full_sess->Create(full_model));

  //
  //
  //
  std::vector<tensorflow::Tensor> input_nodes;
  std::vector<std::string> input_names;

  for (uint32_t n = 0; n < lite_interp->inputs().size(); ++n)
  {
    const TfLiteTensor *tensor = lite_interp->tensor(lite_interp->inputs().at(n));

    input_nodes.emplace_back(tensorflow::DT_FLOAT, asTensorflowShape(tensor));
    input_names.emplace_back(tensor->name);
  }

  assert(input_nodes.size() == input_names.size());
  assert(input_nodes.size() == lite_interp->inputs().size());

  std::vector<std::string> output_names;
  std::vector<tensorflow::Tensor> output_nodes;

  for (uint32_t n = 0; n < lite_interp->outputs().size(); ++n)
  {
    const TfLiteTensor *tensor = lite_interp->tensor(lite_interp->outputs().at(n));

    output_names.emplace_back(tensor->name);
  }

  assert(output_names.size() == lite_interp->outputs().size());
  // output_nodes will be initialized after Tensorflow Session run
  assert(output_nodes.size() == 0);

  //
  // Prepare inference
  //
  lite_sess->prepare();

  // TODO Feed Inputs (for both Tensorflow and Tensorflow Lite)
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;

  for (uint32_t n = 0; n < input_nodes.size(); ++n)
  {
    inputs.emplace_back(input_names.at(0), input_nodes.at(0));
  }

  //
  // Run inference
  //
  TF_ENSURE(full_sess->Run(inputs, output_names, {}, &output_nodes));

  lite_sess->run();

  //
  // Compare Output
  //
  auto equals = [](float lhs, float rhs) {
    // TODO Allow users to set tolerance
    if (nnfw::misc::fp32::absolute_epsilon_equal(lhs, rhs))
    {
      return true;
    }

    return nnfw::misc::fp32::epsilon_equal(lhs, rhs);
  };

  const uint32_t output_count = output_names.size();

  bool matched = true;

  for (uint32_t n = 0; n < output_count; ++n)
  {
    const TfLiteTensor *tensor = lite_interp->tensor(lite_interp->outputs().at(n));

    // TODO Compare shape

    const auto element_count = count_elements(tensor);

    std::cout << "Compare output #" << n << "(" << tensor->name << ", " << element_count
              << " elements)" << std::endl;
    for (uint32_t index = 0; index < element_count; ++index)
    {
      const auto full_value = output_nodes.at(n).flat<float>().data()[index];
      const auto lite_value = lite_sess->interp()->typed_output_tensor<float>(n)[index];

      if (!equals(full_value, lite_value))
      {
        std::cerr << full_value << " is expected, but " << lite_value << " is obtaeind (at " << n
                  << ":" << index << ")" << std::endl;
        matched = false;
      }
    }
  }

  //
  // Cleanup
  //
  lite_sess->teardown();

  return matched ? 0 : 255;
}
