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

#include "ggma_context.h"
#include "Context.h"
#include <iostream>

extern "C" {

GGMA_STATUS ggma_create_context(ggma_context **context, const char *package_path)
{
  if (context == nullptr)
    return GGMA_STATUS_UNEXPECTED_NULL;
  try
  {
    *context = reinterpret_cast<ggma_context *>(new ggma::Context(package_path));
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Error during context creation" << std::endl;
    *context = nullptr;
    return GGMA_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::exception &e)
  {
    std::cerr << "Error during context initialization : " << e.what() << std::endl;
    *context = nullptr;
    return GGMA_STATUS_ERROR;
  }
  return GGMA_STATUS_NO_ERROR;
}

GGMA_STATUS ggma_free_context(ggma_context *context)
{
  delete reinterpret_cast<ggma::Context *>(context);
  return GGMA_STATUS_NO_ERROR;
}

} // extern "C"
