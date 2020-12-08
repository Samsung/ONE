/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CONTROLFLOW_EXTERNAL_CONTEXT_H__
#define __ONERT_BACKEND_CONTROLFLOW_EXTERNAL_CONTEXT_H__

#include <util/ConfigSource.h>

#include <ruy/context.h>
#include <ruy/context_get_ctx.h>
#include <ruy/ctx.h>
#include <ruy/tune.h>

namespace
{
const int kDefaultNumThreadpoolThreads = 1;
}

namespace onert
{
namespace backend
{
namespace controlflow
{

// TODO Unify this with cpu::ExternalContext
class ExternalContext
{
public:
  ExternalContext() : _ruy_context(std::make_unique<ruy::Context>())
  {
    setMaxNumThreads(onert::util::getConfigInt(onert::util::config::RUY_THREADS));
    initPerThreadState();
  }

  void setMaxNumThreads(int max_num_threads)
  {
    const int target_num_threads =
      max_num_threads > -1 ? max_num_threads : kDefaultNumThreadpoolThreads;
    _ruy_context->set_max_num_threads(target_num_threads);
  }

  ruy::Context *ruy_context() const { return _ruy_context.get(); }

private:
  void initPerThreadState()
  {
    // Initialize per-thread state.
    const int thread_count = _ruy_context->max_num_threads();
    auto ctx = ruy::get_ctx(_ruy_context.get());
    ctx->EnsureThreadSpecificResources(thread_count);
    for (int i = 0; i < thread_count; i++)
    {
      ctx->GetThreadSpecificTuningResolver(i)->SetTuning(ctx->explicit_tuning());
    }
  }

private:
  const std::unique_ptr<ruy::Context> _ruy_context;
};

} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_EXTERNAL_CONTEXT_H__
