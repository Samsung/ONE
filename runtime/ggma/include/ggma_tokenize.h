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
 * @file  ggma_tokenize.h
 * @brief This file defines the GGMA Tokenizer interface.
 */
#ifndef __GGMA_TOKENIZE_H__
#define __GGMA_TOKENIZE_H__

#include "ggma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tokenizes an input text string into a sequence of token IDs.
 *
 * This function uses the vocabulary from the provided GGMA package to convert
 * the input text into a series of numerical token IDs.
 *
 * @param[in]  pkg           The GGMA package containing the vocabulary for tokenization.
 * @param[in]  text          The null-terminated text string to be tokenized.
 * @param[in]  text_len      The length of the text in bytes. If the text is null-terminated,
 *                           this can be 0 and the length will be determined internally.
 * @param[out] tokens        A pointer to the output buffer where the generated token IDs will be
 * stored.
 * @param[in]  n_tokens_max  The maximum number of tokens (elements) that the @p tokens buffer can
 * hold.
 * @param[out] n_tokens      A pointer to a variable that will receive the actual number of
 *                           tokens written to the @p tokens buffer.
 * @return     @c GGMA_STATUS_NO_ERROR if successful, or an appropriate error code on failure
 *             (e.g., GGMA_STATUS_UNEXPECTED_NULL if @p pkg or @p text is NULL,
 *             or if the output buffer is too small).
 */
GGMA_STATUS ggma_tokenize(const struct ggma_package *pkg, const char *text, size_t text_len,
                          ggma_token *tokens, size_t n_tokens_max, size_t *n_tokens);

#ifdef __cplusplus
}
#endif

#endif // __GGMA_TOKENIZE_H__
