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

/**
 * @file  ggma_generate.h
 * @brief This file defines the GGMA text generation API.
 */
#ifndef __GGMA_GENERATE_H__
#define __GGMA_GENERATE_H__

#include "ggma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
struct ggma_context;

/**
 * @brief Generates a sequence of tokens based on the provided prompt tokens.
 *
 * This function performs the core inference step, taking an initial sequence of
 * prompt tokens and generating new tokens autoregressively.
 *
 * @param[in]    context  The GGMA context to use for generation.
 * @param[inout] tokens   An array of input prompt tokens. The generated tokens will
 *                        be placed in this buffer
 * @param[in]  n_tokens   The number of tokens in the input @p tokens array. This also
 *                        often specifies the maximum number of tokens to generate.
 * @param[in]  n_tokens_max The maximum number of tokens that the @p tokens buffer can hold.
 * @param[out] n_tokens_out A pointer to a variable that will receive the number of
 *                          element in the @p tokens after generation
 * @return    @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_generate(struct ggma_context *context, ggma_token *tokens, size_t n_tokens,
                          size_t n_tokens_max, size_t *n_tokens_out);

#ifdef __cplusplus
}
#endif

#endif // __GGMA_GENERATE_H__
