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
 * @file  ggma.h
 * @brief This file defines the Generative Model API (GGMA).
 */
#ifndef __GGMA_H__
#define __GGMA_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enumeration of status codes returned by GGMA API functions.
 */
typedef enum
{
  /** The operation completed successfully. */
  GGMA_STATUS_NO_ERROR = 0,
  /**
   * A general error occurred.
   * This code is typically used when no more specific error code is applicable.
   */
  GGMA_STATUS_ERROR = 1,
  /** Unexpected null argument is given. */
  GGMA_STATUS_UNEXPECTED_NULL = 2,
  /** The system ran out of memory. */
  GGMA_STATUS_OUT_OF_MEMORY = 4,
  /** The called API function is deprecated and should no longer be used. */
  GGMA_STATUS_DEPRECATED_API = 6,
} GGMA_STATUS;

/**
 * @brief Opaque handle to a GGMA package.
 *
 * A GGMA package encapsulates all necessary resources for a generative model,
 * including the model itself, default configurations, and vocabulary.
 */
typedef struct ggma_pkg ggma_pkg;

/**
 * @brief Creates a GGMA package from a specified directory path.
 *
 * This function loads the necessary components (model, configuration, vocabulary)
 * from the given directory and initializes a GGMA package handle.
 *
 * @param[out] pkg   Pointer to the package object created from the given path
 * @param[in]  path  Path to the directory containing the GGMA package
 * @return     @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_new_package(ggma_pkg **pkg, const char *path);

/**
 * @brief Frees all resources associated with a GGMA package.
 *
 * After calling {@link ggma_create_session_with_package}, the created session
 * assumes ownership of the GGMA package and will automatically free its resources
 * when the session is closed. Therefore, you must not call this function manually
 * on a package that has been used to create a session, as doing so will lead to
 * a double-free error.
 *
 * @param[in] pkg     The GGMA package to free. This handle will be invalid after the call.
 * @return     @c GGMA_STATUS_NO_ERROR if successful, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_free_package(ggma_pkg *pkg);

/* [G] TODO: PyTorch returns token IDs as int64. However, int32_t is likely sufficient. */
typedef int64_t ggma_token;

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
GGMA_STATUS ggma_tokenize(const struct ggma_pkg *pkg, const char *text, size_t text_len,
                          ggma_token *tokens, size_t n_tokens_max, size_t *n_tokens);

/**
 * @brief Opaque handle to a GGMA inference session.
 *
 * A GGMA session represents a single instance of a loaded model, ready for inference.
 * It holds the runtime context, including the computation graph, key-value (KV) caches,
 * and other resources necessary for token generation.
 */
typedef struct ggma_session ggma_session;

/**
 * @brief Creates a session from a GGMA package.
 *
 * This function initializes a session for inference, taking ownership of the
 * provided GGMA package. The session will manage the lifecycle of the package.
 * Once the session is no longer needed, it must be closed by calling
 * {@link ggma_close_session}.
 *
 * @param[out]  session A pointer to the variable that will receive the new session handle.
 * @param[in]   pkg     The GGMA package to use for creating the session. The session
 *                      will take ownership of this package.
 * @return      @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_create_session_with_package(ggma_session **session, ggma_pkg *pkg);

/**
 * @brief Closes a GGMA session and releases its resources.
 *
 * This function deallocates all resources associated with the session, including
 * the GGMA package it was created with. After this function returns, the session
 * handle is invalid and must not be used.
 *
 * @param[in] session  The session to close.
 * @return    @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_close_session(ggma_session *session);

/**
 * @brief Generates a sequence of tokens based on the provided prompt tokens.
 *
 * This function performs the core inference step, taking an initial sequence of
 * prompt tokens and generating new tokens autoregressively.
 *
 * @param[in]    session  The GGMA session to use for generation.
 * @param[inout] tokens   An array of input prompt tokens. The generated tokens will
 *                        be placed in this buffer
 * @param[in]  n_tokens   The number of tokens in the input @p tokens array. This also
 *                        often specifies the maximum number of tokens to generate.
 * @param[in]  n_tokens_max The maximum number of tokens that the @p tokens buffer can hold.
 * @param[out] n_tokens_out A pointer to a variable that will receive the number of
 *                          element in the @p tokens after generation
 * @return    @c GGMA_STATUS_NO_ERROR on success, or an appropriate error code on failure.
 */
GGMA_STATUS ggma_generate(ggma_session *session, ggma_token *tokens, size_t n_tokens,
                          size_t n_tokens_max, size_t *n_tokens_out);

GGMA_STATUS ggma_set_config(ggma_session *session, const char *key, const char *value);
GGMA_STATUS ggma_get_config(ggma_session *session, const char *key, char *value, size_t value_size);

#ifdef __cplusplus
}
#endif

#endif
