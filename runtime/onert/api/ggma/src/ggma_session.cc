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

#include "ggma_session.h"

#include <memory>
#include <iostream>
#include <string>

ggma_session::ggma_session(ggma_pkg *pkg) : _pkg(pkg) {}

GGMA_STATUS ggma_session::from_package(ggma_session **session, ggma_pkg *pkg)
{
  if (session == nullptr)
    return GGMA_STATUS_UNEXPECTED_NULL;
  try
  {
    auto new_session = std::unique_ptr<ggma_session>(new ggma_session(pkg));
    *session = new_session.release();
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Error during session creation" << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return GGMA_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during session initialization : " << e.what() << std::endl;
    *session = nullptr; // Set nullptr on error to keep the old behavior
    return GGMA_STATUS_ERROR;
  }
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_session::generate(ggma_token *tokens, size_t n_tokens, size_t n_tokens_max,
                                   size_t *n_tokens_out)
{
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_session::set_config(const char *key, const char *value)
{
  return GGMA_STATUS_NO_ERROR;
}
GGMA_STATUS ggma_session::get_config(const char *key, char *value, size_t value_size)
{
  return GGMA_STATUS_NO_ERROR;
}
