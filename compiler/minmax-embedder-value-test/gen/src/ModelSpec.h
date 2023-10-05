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

#ifndef __MINMAX_EMBEDDER_TEST_MODEL_SPEC_H__
#define __MINMAX_EMBEDDER_TEST_MODEL_SPEC_H__

#include <cstdint>

namespace minmax_embedder_test
{
struct ModelSpec
{
  /** number of model inputs */
  uint32_t n_inputs;
  /** number of operators*/
  uint32_t n_ops;
};
} // end of namespace minmax_embedder_test

#endif // __MINMAX_EMBEDDER_TEST_MODEL_SPEC_H__
