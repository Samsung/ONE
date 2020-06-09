/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflite/RandomTestRunner.h"
#include "tflite/Diff.h"
#include "tflite/TensorLogger.h"
#include "tflite/ext/nnapi_delegate.h"

#include <misc/tensor/IndexIterator.h>
#include <misc/tensor/Object.h>
#include <misc/EnvVar.h>
#include <misc/fp32.h>

#include <cassert>
#include <map>
#include <functional>
#include <iostream>

namespace nnfw
{
namespace tflite
{

using namespace std::placeholders;

int RandomTestRunner::run(const nnfw::tflite::Builder &builder)
{
  auto tfl_interp = builder.build();
  auto nnapi = builder.build();

  tfl_interp->UseNNAPI(false);

  // Allocate Tensors
  tfl_interp->AllocateTensors();
  nnapi->AllocateTensors();

  assert(tfl_interp->inputs() == nnapi->inputs());

  using ::tflite::Interpreter;
  using Initializer = std::function<void(int id, Interpreter *, Interpreter *)>;

  std::map<TfLiteType, Initializer> initializers;
  std::map<TfLiteType, Initializer> reseters;

  // Generate singed 32-bit integer (s32) input
  initializers[kTfLiteInt32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteInt32);
    assert(nnapi->tensor(id)->type == kTfLiteInt32);

    auto tfl_interp_view = nnfw::tflite::TensorView<int32_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<int32_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    int32_t value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             // TODO Generate random values
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
             ++value;
           };
  };

  // Generate singed 32-bit integer (s32) input
  reseters[kTfLiteInt32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteInt32);
    assert(nnapi->tensor(id)->type == kTfLiteInt32);

    auto tfl_interp_view = nnfw::tflite::TensorView<int32_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<int32_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    int32_t value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             // TODO Generate random values
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  initializers[kTfLiteUInt8] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteUInt8);
    assert(nnapi->tensor(id)->type == kTfLiteUInt8);

    auto tfl_interp_view = nnfw::tflite::TensorView<uint8_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<uint8_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<uint8_t (nnfw::misc::RandomGenerator::*)(
        const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
        &nnfw::misc::RandomGenerator::generate<uint8_t>);
    const nnfw::misc::tensor::Object<uint8_t> data(tfl_interp_view.shape(),
                                                   std::bind(fp, _randgen, _1, _2));
    assert(tfl_interp_view.shape() == data.shape());

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             const auto value = data.at(ind);

             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  reseters[kTfLiteUInt8] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteUInt8);
    assert(nnapi->tensor(id)->type == kTfLiteUInt8);

    auto tfl_interp_view = nnfw::tflite::TensorView<uint8_t>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<uint8_t>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<uint8_t (nnfw::misc::RandomGenerator::*)(
        const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
        &nnfw::misc::RandomGenerator::generate<uint8_t>);
    const nnfw::misc::tensor::Object<uint8_t> data(tfl_interp_view.shape(),
                                                   std::bind(fp, _randgen, _1, _2));
    assert(tfl_interp_view.shape() == data.shape());

    uint8_t value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  initializers[kTfLiteFloat32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteFloat32);
    assert(nnapi->tensor(id)->type == kTfLiteFloat32);

    auto tfl_interp_view = nnfw::tflite::TensorView<float>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<float>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<float (nnfw::misc::RandomGenerator::*)(
        const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
        &nnfw::misc::RandomGenerator::generate<float>);
    const nnfw::misc::tensor::Object<float> data(tfl_interp_view.shape(),
                                                 std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             const auto value = data.at(ind);

             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  reseters[kTfLiteFloat32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteFloat32);
    assert(nnapi->tensor(id)->type == kTfLiteFloat32);

    auto tfl_interp_view = nnfw::tflite::TensorView<float>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<float>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<float (nnfw::misc::RandomGenerator::*)(
        const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
        &nnfw::misc::RandomGenerator::generate<float>);
    const nnfw::misc::tensor::Object<float> data(tfl_interp_view.shape(),
                                                 std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    float value = 0;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  initializers[kTfLiteBool] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteBool);
    assert(nnapi->tensor(id)->type == kTfLiteBool);

    auto tfl_interp_view = nnfw::tflite::TensorView<bool>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<bool>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<bool (nnfw::misc::RandomGenerator::*)(
        const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
        &nnfw::misc::RandomGenerator::generate<bool>);
    const nnfw::misc::tensor::Object<bool> data(tfl_interp_view.shape(),
                                                std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             const auto value = data.at(ind);

             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  reseters[kTfLiteBool] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(tfl_interp->tensor(id)->type == kTfLiteBool);
    assert(nnapi->tensor(id)->type == kTfLiteBool);

    auto tfl_interp_view = nnfw::tflite::TensorView<bool>::make(*tfl_interp, id);
    auto nnapi_view = nnfw::tflite::TensorView<bool>::make(*nnapi, id);

    assert(tfl_interp_view.shape() == nnapi_view.shape());

    auto fp = static_cast<bool (nnfw::misc::RandomGenerator::*)(
        const ::nnfw::misc::tensor::Shape &, const ::nnfw::misc::tensor::Index &)>(
        &nnfw::misc::RandomGenerator::generate<bool>);
    const nnfw::misc::tensor::Object<bool> data(tfl_interp_view.shape(),
                                                std::bind(fp, _randgen, _1, _2));

    assert(tfl_interp_view.shape() == data.shape());

    bool value = false;

    nnfw::misc::tensor::iterate(tfl_interp_view.shape())
        << [&](const nnfw::misc::tensor::Index &ind) {
             tfl_interp_view.at(ind) = value;
             nnapi_view.at(ind) = value;
           };
  };

  // Fill IFM with random numbers
  for (const auto id : tfl_interp->inputs())
  {
    assert(tfl_interp->tensor(id)->type == nnapi->tensor(id)->type);

    auto it = initializers.find(tfl_interp->tensor(id)->type);

    if (it == initializers.end())
    {
      throw std::runtime_error{"Not supported input type"};
    }

    it->second(id, tfl_interp.get(), nnapi.get());
  }

  // Fill OFM with 0
  for (const auto id : tfl_interp->outputs())
  {
    assert(tfl_interp->tensor(id)->type == nnapi->tensor(id)->type);

    auto it = reseters.find(tfl_interp->tensor(id)->type);

    if (it == reseters.end())
    {
      throw std::runtime_error{"Not supported input type"};
    }

    it->second(id, tfl_interp.get(), nnapi.get());
  }

  std::cout << "[NNAPI TEST] Run T/F Lite Interpreter without NNAPI" << std::endl;
  tfl_interp->Invoke();

  std::cout << "[NNAPI TEST] Run T/F Lite Interpreter with NNAPI" << std::endl;

  char *env = getenv("UPSTREAM_DELEGATE");

  if (env && !std::string(env).compare("1"))
  {
    nnapi->UseNNAPI(true);
    nnapi->Invoke();
  }
  else
  {
    nnfw::tflite::NNAPIDelegate d;

    // WARNING
    // primary_subgraph: Experimental interface. Return 1st sugbraph
    if (d.BuildGraph(&nnapi.get()->primary_subgraph()))
    {
      throw std::runtime_error{"Failed to BuildGraph"};
    }

    if (d.Invoke(&nnapi.get()->primary_subgraph()))
    {
      throw std::runtime_error{"Failed to BuildGraph"};
    }
  }

  // Compare OFM
  std::cout << "[NNAPI TEST] Compare the result" << std::endl;

  const auto tolerance = _param.tolerance;

  auto equals = [tolerance](float lhs, float rhs) {
    // NOTE Hybrid approach
    // TODO Allow users to set tolerance for absolute_epsilon_equal
    if (nnfw::misc::fp32::absolute_epsilon_equal(lhs, rhs))
    {
      return true;
    }

    return nnfw::misc::fp32::epsilon_equal(lhs, rhs, tolerance);
  };

  nnfw::misc::tensor::Comparator comparator(equals);
  TfLiteInterpMatchApp app(comparator);

  app.verbose() = _param.verbose;

  bool res = app.run(*tfl_interp, *nnapi);

  if (!res)
  {
    return 255;
  }

  std::cout << "[NNAPI TEST] PASSED" << std::endl;

  if (_param.tensor_logging)
    nnfw::tflite::TensorLogger::get().save(_param.log_path, *tfl_interp);

  return 0;
}

RandomTestRunner RandomTestRunner::make(uint32_t seed)
{
  RandomTestParam param;

  param.verbose = nnfw::misc::EnvVar("VERBOSE").asInt(0);
  param.tolerance = nnfw::misc::EnvVar("TOLERANCE").asInt(1);
  param.tensor_logging = nnfw::misc::EnvVar("TENSOR_LOGGING").asBool(false);
  param.log_path = nnfw::misc::EnvVar("TENSOR_LOGGING").asString("tensor_log.txt");

  return RandomTestRunner{seed, param};
}

} // namespace tflite
} // namespace nnfw
