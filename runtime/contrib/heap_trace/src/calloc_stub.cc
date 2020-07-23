/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "trace.h"
#include "function_resolver.h"
#include "memory_pool_for_symbol_searcher_internals.h"

#include <memory>

extern std::unique_ptr<Trace> GlobalTrace;

extern "C"
{

  void *calloc(size_t number_of_elements, size_t size_of_one_element) noexcept
  {
    static auto isOriginalFunctionCallSuccessful = [](void *result) -> bool { return result; };

    if (isCurrentAllocationForSymbolSearcherInternalUsage())
    {
      return MemoryPoolForSymbolSearcherInternals{}.allocate(number_of_elements *
                                                             size_of_one_element);
    }

    static auto originalFunction = findFunctionByName<void *, size_t, size_t>("calloc");
    void *result = originalFunction(number_of_elements, size_of_one_element);
    if (isOriginalFunctionCallSuccessful(result) && !Trace::Guard{}.isActive())
    {
      GlobalTrace->logAllocationEvent(result, number_of_elements * size_of_one_element);
    }

    return result;
  }
}
