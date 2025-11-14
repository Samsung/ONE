/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GGML_EXTERNAL_CONTEXT_H__
#define __ONERT_BACKEND_GGML_EXTERNAL_CONTEXT_H__

#include <util/ConfigSource.h>
#include <ggml.h>

#include <memory>

namespace onert::backend::ggml
{

class ExternalContext
{
public:
  ExternalContext();

public:
  void setMaxNumThreads(int max_num_threads);

  int32_t maxNumThreads() const { return _max_num_threads; }

private:
  int32_t _max_num_threads;
  std::unique_ptr<ggml_context, decltype(&ggml_free)> _ggml_context{nullptr, &ggml_free};
};

} // namespace onert::backend::ggml

#endif // __ONERT_BACKEND_GGML_EXTERNAL_CONTEXT_H__
