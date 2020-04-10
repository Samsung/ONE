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

#ifndef __NNFW_CKER_RUY_RUY_SUPPORT_H__
#define __NNFW_CKER_RUY_RUY_SUPPORT_H__

#include <ruy/context.h>

namespace
{
const int kDefaultNumThreadpoolThreads = 1;
}

namespace nnfw
{
namespace cker
{
namespace ruy_support
{

struct RuyContext
{
public:
  RuyContext() : ruy_context_(new ruy::Context)
  {
    SetMaxNumThreads(kDefaultNumThreadpoolThreads);
#ifdef TFLITE_WITH_RUY_GEMV
    ruy_context_->cache_policy = ruy::kCacheLHSOnNarrowMul;
#endif
  };

  ruy::Context *ruy_context() const { return ruy_context_.get(); }

  static inline RuyContext &GetRuyContext()
  {
    static RuyContext instance;
    return instance;
  }

  void SetMaxNumThreads(int max_num_threads)
  {
    const int target_num_threads =
        max_num_threads > -1 ? max_num_threads : kDefaultNumThreadpoolThreads;
    ruy_context_->max_num_threads = target_num_threads;
  }

private:
  const std::unique_ptr<ruy::Context> ruy_context_;
};

inline ruy::Context *GetRuyContext()
{
  auto &ctx = RuyContext::GetRuyContext();
  return ctx.ruy_context.get();
}

} // namespace ruy_support
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_RUY_RUY_SUPPORT_H__
