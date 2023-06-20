/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_TRAIN_RANDOMGEN_H__
#define __ONERT_TRAIN_RANDOMGEN_H__

#include <string>
#include <vector>

#include "nnfw.h"
#include "allocation.h"

struct nnfw_session;

namespace onert_train
{
// class RandomGenerator
// {
// public:
//   RandomGenerator(nnfw_session *sess) : session_(sess) {}
//   void generate(std::vector<Allocation> &inputs, int num_inputs, int data_length);

// private:
//   nnfw_session *session_;
// };
} // namespace onert_train

#endif // __ONERT_TRAIN_RANDOMGEN_H__
