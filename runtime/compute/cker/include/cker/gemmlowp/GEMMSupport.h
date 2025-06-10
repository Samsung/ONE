/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_GEMMLOWP_GEMM_SUPPORT_H__
#define __NNFW_CKER_GEMMLOWP_GEMM_SUPPORT_H__

#include <public/gemmlowp.h>

#include <memory>
#include <thread>

namespace nnfw
{
namespace cker
{
namespace gemm_support
{

struct GemmContext
{
  std::unique_ptr<gemmlowp::GemmContext> gemm_context;
  constexpr static int default_num_threadpool_threads = 4;

  GemmContext()
  {
    int num_threads = std::thread::hardware_concurrency() / 2;
    if (num_threads == 0)
    {
      num_threads = default_num_threadpool_threads;
    }

    gemm_context.reset(new gemmlowp::GemmContext());
    gemm_context->set_max_num_threads(num_threads);
  }

  static inline GemmContext &GetGemmLowpContext()
  {
    static GemmContext instance;
    return instance;
  }
};

inline gemmlowp::GemmContext *GetGemmLowpContext()
{
  auto &ctx = GemmContext::GetGemmLowpContext();
  return ctx.gemm_context.get();
}

} // namespace gemm_support
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_GEMMLOWP_GEMM_SUPPORT_H__
