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

#include "tflite/RandomInputInitializer.h"
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

class CopyInputInitializer
{
public:
  CopyInputInitializer(::tflite::Interpreter &from) : _from{from}
  {
    // DO NOTHING
  }

  void run(::tflite::Interpreter &interp)
  {
    for (const auto &tensor_idx : interp.inputs())
    {
      TfLiteTensor *tensor = interp.tensor(tensor_idx);
      if (tensor->type == kTfLiteInt32)
      {
        setValue<int32_t>(interp, tensor_idx);
      }
      else if (tensor->type == kTfLiteUInt8)
      {
        setValue<uint8_t>(interp, tensor_idx);
      }
      else if (tensor->type == kTfLiteInt8)
      {
        setValue<int8_t>(interp, tensor_idx);
      }
      else if (tensor->type == kTfLiteBool)
      {
        setValue<bool>(interp, tensor_idx);
      }
      else
      {
        assert(tensor->type == kTfLiteFloat32);

        setValue<float>(interp, tensor_idx);
      }
    }
  }

private:
  template <typename T> void setValue(::tflite::Interpreter &interp, int tensor_idx)
  {
    auto tensor_from_view = nnfw::tflite::TensorView<T>::make(_from, tensor_idx);
    auto tensor_to_view = nnfw::tflite::TensorView<T>::make(interp, tensor_idx);

    nnfw::misc::tensor::iterate(tensor_from_view.shape())
      << [&](const nnfw::misc::tensor::Index &ind) {
           tensor_to_view.at(ind) = tensor_from_view.at(ind);
         };
  }

private:
  ::tflite::Interpreter &_from;
};

void RandomTestRunner::compile(const nnfw::tflite::Builder &builder)
{
  _tfl_interp = builder.build();
  _nnapi = builder.build();

  _tfl_interp->UseNNAPI(false);

  // Allocate Tensors
  _tfl_interp->AllocateTensors();
  _nnapi->AllocateTensors();

  assert(_tfl_interp->inputs() == _nnapi->inputs());

  using ::tflite::Interpreter;
  using Initializer = std::function<void(int id, Interpreter *, Interpreter *)>;

  std::map<TfLiteType, Initializer> reseters;

  // Generate singed 32-bit integer (s32) input
  reseters[kTfLiteInt32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(_tfl_interp->tensor(id)->type == kTfLiteInt32);
    assert(_nnapi->tensor(id)->type == kTfLiteInt32);

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

  reseters[kTfLiteUInt8] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(_tfl_interp->tensor(id)->type == kTfLiteUInt8);
    assert(_nnapi->tensor(id)->type == kTfLiteUInt8);

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

  reseters[kTfLiteFloat32] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(_tfl_interp->tensor(id)->type == kTfLiteFloat32);
    assert(_nnapi->tensor(id)->type == kTfLiteFloat32);

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

  reseters[kTfLiteBool] = [&](int id, Interpreter *tfl_interp, Interpreter *nnapi) {
    assert(_tfl_interp->tensor(id)->type == kTfLiteBool);
    assert(_nnapi->tensor(id)->type == kTfLiteBool);

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

  RandomInputInitializer initializer{_randgen};
  initializer.run(*(_tfl_interp.get()));

  CopyInputInitializer copy_initializer{*(_tfl_interp.get())};
  copy_initializer.run(*(_nnapi.get()));

  // Fill OFM with 0
  for (const auto id : _tfl_interp->outputs())
  {
    assert(_tfl_interp->tensor(id)->type == _nnapi->tensor(id)->type);

    auto it = reseters.find(_tfl_interp->tensor(id)->type);

    if (it == reseters.end())
    {
      throw std::runtime_error{"Not supported input type"};
    }

    it->second(id, _tfl_interp.get(), _nnapi.get());
  }
}

int RandomTestRunner::run(size_t running_count)
{
  std::cout << "[NNAPI TEST] Run T/F Lite Interpreter without NNAPI" << std::endl;
  _tfl_interp->Invoke();

  nnfw::tflite::NNAPIDelegate d;

  for (size_t i = 1; i <= running_count; ++i)
  {
    std::cout << "[NNAPI TEST #" << i << "] Run T/F Lite Interpreter with NNAPI" << std::endl;

    char *env = getenv("UPSTREAM_DELEGATE");

    if (env && !std::string(env).compare("1"))
    {
      _nnapi->UseNNAPI(true);
      _nnapi->Invoke();
    }
    else
    {
      // WARNING
      // primary_subgraph: Experimental interface. Return 1st sugbraph
      // Invoke() will call BuildGraph() internally
      if (d.Invoke(&_nnapi.get()->primary_subgraph()))
      {
        throw std::runtime_error{"Failed to BuildGraph"};
      }
    }

    // Compare OFM
    std::cout << "[NNAPI TEST #" << i << "] Compare the result" << std::endl;

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

    nnfw::misc::tensor::Comparator<float> comparator(equals);
    TfLiteInterpMatchApp app(comparator);

    app.verbose() = _param.verbose;

    bool res = app.run(*_tfl_interp, *_nnapi);

    if (!res)
    {
      return 255;
    }

    std::cout << "[NNAPI TEST #" << i << "] PASSED" << std::endl << std::endl;

    if (_param.tensor_logging)
      nnfw::tflite::TensorLogger::get().save(_param.log_path, *_tfl_interp);
  }

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
