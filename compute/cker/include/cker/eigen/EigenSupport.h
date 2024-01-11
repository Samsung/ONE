/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_EIGEN_EIGEN_SUPPORT_H__
#define __NNFW_CKER_EIGEN_EIGEN_SUPPORT_H__

// #if defined(CKER_OPTIMIZED_EIGEN)

#include <Eigen/Core>
#include <thread>
#include "cker/eigen/eigen_spatial_convolutions.h"

#ifdef EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/ThreadPool>
#endif

namespace nnfw
{
namespace cker
{
namespace eigen_support
{

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
  EigenMatrix;
typedef Eigen::TensorMap<Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
  ConstEigenMatrix;

typedef Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
  EigenTensor;
typedef Eigen::TensorMap<Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
                         Eigen::Aligned>
  ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T> struct MatMulConvFunctor
{
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(const Device &d, EigenMatrix out, ConstEigenMatrix in0, ConstEigenMatrix in1,
                  const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> &dim_pair)
  {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

// We have a single global threadpool for all convolution operations. This means
// that inferences started from different threads may block each other, but
// since the underlying resource of CPU cores should be consumed by the
// operations anyway, it shouldn't affect overall performance.
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface
{
public:
  // Takes ownership of 'pool'
  explicit EigenThreadPoolWrapper(Eigen::ThreadPool *pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}

  void Schedule(std::function<void()> fn) override { pool_->Schedule(std::move(fn)); }
  int NumThreads() const override { return pool_->NumThreads(); }
  int CurrentThreadId() const override { return pool_->CurrentThreadId(); }

private:
  std::unique_ptr<Eigen::ThreadPool> pool_;
};

struct EigenContext
{
  constexpr static int default_num_threadpool_threads = 4;
  std::unique_ptr<Eigen::ThreadPoolInterface> thread_pool_wrapper;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;

  EigenContext()
  {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
    {
      num_threads = default_num_threadpool_threads;
    }
    device.reset(); // destroy before we invalidate the thread pool
    thread_pool_wrapper.reset(new EigenThreadPoolWrapper(new Eigen::ThreadPool(num_threads)));
    device.reset(new Eigen::ThreadPoolDevice(thread_pool_wrapper.get(), num_threads));
  }

  static inline EigenContext &GetEigenContext()
  {
    static EigenContext instance;
    return instance;
  }
};

inline const Eigen::ThreadPoolDevice *GetThreadPoolDevice()
{
  auto &ctx = EigenContext::GetEigenContext();
  return ctx.device.get();
}

} // namespace eigen_support
} // namespace cker
} // namespace nnfw

// #endif // defined(CKER_OPTIMIZED_EIGEN)

#endif // __NNFW_CKER_EIGEN_EIGEN_SUPPORT_H__
