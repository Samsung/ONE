/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Job.h"

#include <cassert>

#include "util/logging.h"

namespace onert
{
namespace exec
{

Job::Job(uint32_t index, FunctionSequence *fn_seq) : _index{index}, _fn_seq{fn_seq} {}

void Job::run() { _fn_seq->run(); }

} // namespace exec
} // namespace onert
