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

/**
 * @file TransposeConv benchmark with various algorithms
 */

#include <nonius/nonius.h++>

#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>

#include <cstdint>
#include <cassert>
#include <stdexcept>

#include "acl_common/Utils.h"

using namespace arm_compute;
using namespace kbenchmark::kernels::acl_common;

//
// Helpers
//
namespace
{

enum Layout
{
  NCHW,
  NHWC
};

TensorInfo make_info(uint32_t N)
{
  TensorShape shape{N};
  return TensorInfo{shape, 1, DataType::F32};
}

template <enum Layout> TensorInfo make_info(uint32_t N, uint32_t C, uint32_t H, uint32_t W);

template <> TensorInfo make_info<NCHW>(uint32_t N, uint32_t C, uint32_t H, uint32_t W)
{
  TensorShape shape{W, H, C, N};
  TensorInfo info{shape, 1, DataType::F32};
  info.set_data_layout(DataLayout::NCHW);
  return info;
}

template <> TensorInfo make_info<NHWC>(uint32_t N, uint32_t C, uint32_t H, uint32_t W)
{
  TensorShape shape{C, W, H, N};
  TensorInfo info{shape, 1, DataType::F32};
  info.set_data_layout(DataLayout::NHWC);
  return info;
}

inline void check(const Status &status)
{
  if (!status)
  {
    std::cerr << status.error_description() << std::endl;
    throw std::runtime_error{"ERROR"};
  }
}

inline bool is_odd(uint32_t n) { return (n % 2 != 0) ? true : false; }

} // namespace

//
// Benchmark Parameters
//
NONIUS_PARAM(BATCH, 1);

NONIUS_PARAM(IFM_C, 3);
NONIUS_PARAM(IFM_H, 244);
NONIUS_PARAM(IFM_W, 244);

NONIUS_PARAM(OFM_C, 3);
NONIUS_PARAM(OFM_H, 244);
NONIUS_PARAM(OFM_W, 244);

NONIUS_PARAM(KER_H, 3);
NONIUS_PARAM(KER_W, 3);

NONIUS_PARAM(STRIDE_H, 1);
NONIUS_PARAM(STRIDE_W, 1);

NONIUS_PARAM(PADDING, std::string{"SAME"})

//
// Configuration Helpers
//
namespace
{

struct Configuration
{
  uint32_t ifm_N;
  uint32_t ifm_C;
  uint32_t ifm_H;
  uint32_t ifm_W;

  uint32_t ofm_N;
  uint32_t ofm_C;
  uint32_t ofm_H;
  uint32_t ofm_W;

  uint32_t ker_N;
  uint32_t ker_C;
  uint32_t ker_H;
  uint32_t ker_W;

  uint32_t vertical_stride;
  uint32_t horizontal_stride;

  PadStrideInfo deconv_info;

  uint32_t inner_border_right;
  uint32_t inner_border_top;

  Configuration(nonius::chronometer meter)
  {
    ifm_N = meter.param<BATCH>();
    ifm_C = meter.param<IFM_C>();
    ifm_H = meter.param<IFM_H>();
    ifm_W = meter.param<IFM_W>();

    ofm_N = meter.param<BATCH>();
    ofm_C = meter.param<OFM_C>();
    ofm_H = meter.param<OFM_H>();
    ofm_W = meter.param<OFM_W>();

    ker_N = meter.param<OFM_C>();
    ker_C = meter.param<IFM_C>();
    ker_H = meter.param<KER_H>();
    ker_W = meter.param<KER_W>();

    vertical_stride = meter.param<STRIDE_H>();
    horizontal_stride = meter.param<STRIDE_W>();

    // NOTE The padding calculation formula of TransposeConv is opposite to Conv.
    //      So the location of ifm and ofm is changed.
    auto padding_info = calculatePadding(meter.param<PADDING>(), ofm_H, ofm_W, ifm_H, ifm_W,
                                         vertical_stride, horizontal_stride, ker_H, ker_W);

    inner_border_right = padding_info.right - padding_info.left;
    inner_border_top = padding_info.bottom - padding_info.top;

    padding_info.left = padding_info.right;
    padding_info.top = padding_info.bottom;

    deconv_info = asPadStrideInfo(padding_info, vertical_stride, horizontal_stride);
  }

  template <Layout L> TensorInfo src_info() const
  {
    return make_info<L>(ifm_N, ifm_C, ifm_H, ifm_W);
  }
  template <Layout L> TensorInfo dst_info() const
  {
    return make_info<L>(ofm_N, ofm_C, ofm_H, ofm_W);
  }
  template <Layout L> TensorInfo ker_info() const
  {
    return make_info<L>(ker_N, ker_C, ker_H, ker_W);
  }
  TensorInfo bias_info(void) const { return make_info(ker_N); }
};

} // namespace

//
// Benchmark Implementations
//
namespace
{

inline nonius::benchmark_registry &local_benchmark_registry()
{
  static nonius::benchmark_registry registry;
  return registry;
}

} // namespace

#define NONIUS_LOCAL_BENCHMARK(name, ...)                                                          \
  namespace                                                                                        \
  {                                                                                                \
  static ::nonius::benchmark_registrar                                                             \
    NONIUS_DETAIL_UNIQUE_NAME(benchmark_registrar)(local_benchmark_registry(), name, __VA_ARGS__); \
  }

NONIUS_LOCAL_BENCHMARK("NEDeconvolutionLayer_NCHW", [](nonius::chronometer meter) {
  NEDeconvolutionLayer deconv;

  // Configure
  Configuration p{meter};

  Tensor src_tensor{};
  Tensor dst_tensor{};
  Tensor ker_tensor{};

  src_tensor.allocator()->init(p.src_info<NCHW>());
  dst_tensor.allocator()->init(p.dst_info<NCHW>());
  ker_tensor.allocator()->init(p.ker_info<NCHW>());

  try
  {
    check(deconv.validate(src_tensor.info(), ker_tensor.info(), nullptr, dst_tensor.info(),
                          p.deconv_info, p.inner_border_right, p.inner_border_top));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  deconv.configure(&src_tensor, &ker_tensor, nullptr, &dst_tensor, p.deconv_info,
                   p.inner_border_right, p.inner_border_top);

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run!
  meter.measure([&](int) { deconv.run(); });
})

NONIUS_LOCAL_BENCHMARK("NEDeconvolutionLayer_NHWC", [](nonius::chronometer meter) {
  NEDeconvolutionLayer deconv;

  // Configure
  Configuration p{meter};

  Tensor src_tensor{};
  Tensor dst_tensor{};
  Tensor ker_tensor{};

  src_tensor.allocator()->init(p.src_info<NHWC>());
  dst_tensor.allocator()->init(p.dst_info<NHWC>());
  ker_tensor.allocator()->init(p.ker_info<NHWC>());

  try
  {
    check(deconv.validate(src_tensor.info(), ker_tensor.info(), nullptr, dst_tensor.info(),
                          p.deconv_info, p.inner_border_right, p.inner_border_top));
  }
  catch (...)
  {
    meter.measure([&](int) {
      // DO NOTHING
      volatile int x = 0;
      return x;
    });
    return;
  }

  deconv.configure(&src_tensor, &ker_tensor, nullptr, &dst_tensor, p.deconv_info,
                   p.inner_border_right, p.inner_border_top);

  src_tensor.allocator()->allocate();
  ker_tensor.allocator()->allocate();
  dst_tensor.allocator()->allocate();

  // Run!
  meter.measure([&](int) { deconv.run(); });
})

extern "C" nonius::benchmark_registry &benchmark_functions(void)
{
  return local_benchmark_registry();
}
