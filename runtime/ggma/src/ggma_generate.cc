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

#include "ggma_generate.h"

#include "Context.h"
#include "Macro.h"

GGMA_STATUS ggma_generate(ggma_context *context, ggma_token *tokens, size_t n_tokens,
                          size_t n_tokens_max, size_t *n_tokens_out)
{
  GGMA_RETURN_ERROR_IF_NULL(context);
  return reinterpret_cast<ggma::Context *>(context)->generate(tokens, n_tokens, n_tokens_max,
                                                              n_tokens_out);
}
