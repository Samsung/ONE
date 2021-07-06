/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_OPENCL_INFERENCE_CONTEXT_H__
#define __ONERT_BACKEND_GPU_CL_OPENCL_INFERENCE_CONTEXT_H__

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>

#include "Buffer.h"
#include "ClCommandQueue.h"
#include "Environment.h"
#include "GpuObject.h"
#include "kernels/GpuOperation.h"
#include "ModelHints.h"
#include "OpenclWrapper.h"
#include "Precision.h"
#include "TensorType.h"
#include "Model.h"
#include "InternalTensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

struct CLNode
{
  std::unique_ptr<GPUOperation> operation;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;

  // Mostly for debug purposes.
  std::string name;

  CLNode() = default;

  CLNode(CLNode &&node);
  CLNode &operator=(CLNode &&node);
  CLNode(const CLNode &) = delete;
  CLNode &operator=(const CLNode &) = delete;
};

class InferenceContext
{
public:
  struct CreateInferenceInfo
  {
    CalculationsPrecision precision;
    TensorStorageType storage_type;
    ModelHints hints;
  };

  struct DummyTensor
  {
    BHWC shape;
    TensorDescriptor descriptor;

    bool operator==(const DummyTensor &b) const
    {
      return shape == b.shape && descriptor == b.descriptor;
    }
  };

  class TensorReserver
  {
  public:
    ValueId Add(const std::shared_ptr<DummyTensor> dummy)
    {
      reservations_[next_] = std::move(dummy);
      return next_++;
    }
    void Add(ValueId id, const std::shared_ptr<DummyTensor> dummy)
    {
      reservations_[id] = std::move(dummy);
    }
    void SetNext(ValueId id) { next_ = id; }
    bool HaveTensor(ValueId id) { return reservations_.find(id) != reservations_.end(); }
    std::shared_ptr<DummyTensor> Get(ValueId id) { return reservations_[id]; }

    std::vector<std::pair<ValueId, TensorDescriptor>> GetTensorDescs() const
    {
      std::vector<std::pair<ValueId, TensorDescriptor>> result;
      for (auto &v : reservations_)
      {
        TensorDescriptor desc = v.second->descriptor;
        desc.shape.b = v.second->shape.b;
        desc.shape.h = v.second->shape.h;
        desc.shape.w = v.second->shape.w;
        desc.shape.d = 1;
        desc.shape.c = v.second->shape.c;
        result.push_back({v.first, desc});
      }
      return result;
    }

    void Add(const std::vector<std::pair<ValueId, TensorDescriptor>> &tensors)
    {
      for (auto &v : tensors)
      {
        auto dummy = std::make_shared<DummyTensor>();
        dummy->descriptor = v.second;
        dummy->shape.b = v.second.shape.b;
        dummy->shape.h = v.second.shape.h;
        dummy->shape.w = v.second.shape.w;
        dummy->shape.c = v.second.shape.c;
        Add(v.first, dummy);
      }
    }

  private:
    std::unordered_map<ValueId, std::shared_ptr<DummyTensor>> reservations_;
    ValueId next_ = 0;
  };

private:
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_OPENCL_INFERENCE_CONTEXT_H__
