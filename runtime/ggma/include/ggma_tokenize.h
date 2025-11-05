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
 * @file  ggma_Tokenizer.h
 * @brief This file defines the GGMA Tokenizer interface.
 */
#ifndef __GGMA_GGMA_TOKENIZE_H__
#define __GGMA_GGMA_TOKENIZE_H__

#include "ggma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a GGMA tokenizer.
 *
 * A GGMA tokenizer encapsulates all necessary components for text tokenization,
 * including the tokenizer model and vocabulary.
 */
typedef struct ggma_tokenizer ggma_tokenizer;

/**
 * @brief Creates a GGMA tokenizer from a specified tokenizer path.
 *
 * This function loads the necessary tokenizer components from the given tokenizer path
 * and initializes a GGMA tokenizer handle.
 *
 * @param[out] tokenizer     Pointer to the tokenizer object created from the given path
 * @param[in]  tokenizer_path The path to the directory containing the tokenizer model and
 * vocabulary
 * @return     @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure
 *             (e.g., GGMA_STATUS_UNEXPECTED_NULL if @p tokenizer_path or @p tokenizer is NULL,
 *             or if the tokenizer cannot be created).
 */
GGMA_STATUS ggma_create_tokenizer(ggma_tokenizer **tokenizer, const char *tokenizer_path);

/**
 * @brief Frees all resources associated with a GGMA tokenizer.
 *
 * @param[in] tokenizer     The GGMA tokenizer to free. This handle will be invalid after the call.
 * @return     @c GGMA_STATUS_NO_ERROR if successful, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_free_tokenizer(ggma_tokenizer *tokenizer);

/**
 * @brief Tokenizes an input text string into a sequence of token IDs.
 *
 * This function uses the vocabulary from the created tokenizer to convert
 * the input text into a series of numerical token IDs.
 *
 * @param[in]  tokenizer     The GGMA tokenizer handle for tokenization.
 * @param[in]  text          The null-terminated text string to be tokenized.
 * @param[in]  text_len      The length of the text in bytes. If the text is null-terminated,
 *                           this can be 0 and the length will be determined internally.
 * @param[out] tokens        Output buffer for generated token IDs.
 * @param[in]  n_tokens_max  Maximum number of tokens the @p tokens buffer can hold.
 * @param[out] n_tokens    A pointer to a variable that will receive the actual number of
 *                         tokens written to the @p tokens buffer.
 * @return     @c GGMA_STATUS_NO_ERROR if successful, or an appropriate error code on failure
 *             (e.g., GGMA_STATUS_UNEXPECTED_NULL if @p tokenizer or @p text is NULL,
 *             or if the output buffer is too small).
 */
GGMA_STATUS ggma_tokenize(const ggma_tokenizer *tokenizer, const char *text, size_t text_len,
                          int32_t *tokens, size_t n_tokens_max, size_t *n_tokens);

/**
 * @brief Detokenizes a sequence of token IDs back into a text string.
 *
 * This function uses the vocabulary from the created tokenizer to convert
 * the sequence of token IDs back into a human-readable text string.
 *
 * @param[in]  tokenizer  The GGMA tokenizer handle for detokenization.
 * @param[in]  tokens     A pointer to the input buffer containing the token IDs to be detokenized.
 * @param[in]  n_tokens   The number of tokens in the @p tokens buffer.
 * @param[out] text       A pointer to the output buffer where the detokenized text will be stored.
 * @param[in]  text_len   The maximum size of the @p text buffer in bytes.
 * @return     @c GGMA_STATUS_NO_ERROR if successful, or an appropriate error code on failure
 *             (e.g., GGMA_STATUS_UNEXPECTED_NULL if @p tokenizer or @p tokens is NULL,
 *             or if the output buffer is too small).
 */
GGMA_STATUS ggma_detokenize(const ggma_tokenizer *tokenizer, const int32_t *tokens, size_t n_tokens,
                            char *text, size_t text_len);

#ifdef __cplusplus
}
#endif

#endif // __GGMA_GGMA_TOKENIZE_H__
