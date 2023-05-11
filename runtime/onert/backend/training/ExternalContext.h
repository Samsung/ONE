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

#ifndef __ONERT_BACKEND_TRAINING_EXTERNAL_CONTEXT_H__
#define __ONERT_BACKEND_TRAINING_EXTERNAL_CONTEXT_H__

#include <util/ConfigSource.h>
#include <ruy/context.h>

#include <memory>

namespace onert
{
namespace backend
{
namespace training
{

class ExternalContext
{
private:
  static const int kDefaultNumThreadpoolThreads = 1;

public:
  ExternalContext() : _ruy_context(new ruy::Context)
  {
    setMaxNumThreads(onert::util::getConfigInt(onert::util::config::RUY_THREADS));
  }

  void setMaxNumThreads(int max_num_threads)
  {
    const int target_num_threads =
      max_num_threads > -1 ? max_num_threads : kDefaultNumThreadpoolThreads;
    _ruy_context->set_max_num_threads(target_num_threads);
  }

  ruy::Context *ruy_context() const { return _ruy_context.get(); }

private:
  const std::unique_ptr<ruy::Context> _ruy_context;
};

} // namespace training
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAINING_EXTERNAL_CONTEXT_H__
