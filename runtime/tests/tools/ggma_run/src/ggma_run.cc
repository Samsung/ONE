/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "args.h"
#include "ggma_api.h"
#include "ggma_macro.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(const int argc, char **argv)
{
  using namespace ggma_run;

  try
  {
    Args args(argc, argv);

    ggma_package *pkg = nullptr;
    GGMA_ENSURE(ggma_create_package(&pkg, args.packagePath().c_str()));

    std::string prompt = "Lily picked up a flower.";
    constexpr size_t n_tokens_max = 32;
    ggma_token tokens[n_tokens_max];
    size_t n_tokens;
    GGMA_ENSURE(ggma_tokenize(pkg, prompt.c_str(), prompt.size(), tokens, n_tokens_max, &n_tokens));

    ggma_context *context = nullptr;
    GGMA_ENSURE(ggma_create_context(&context, pkg));

    size_t n_predict = 10;
    GGMA_ENSURE(ggma_generate(context, tokens, n_tokens, n_tokens_max, &n_predict));

    // Output generated token IDs
    std::cout << "prompt: " << prompt << std::endl;
    std::cout << "generated: { ";
    for (size_t i = n_tokens; i < n_tokens + n_predict; ++i)
    {
      std::cout << tokens[i];
      if (i < n_tokens + n_predict - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << " }" << std::endl;

    GGMA_ENSURE(ggma_free_context(context));
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error: " << e.what() << std::endl;
    exit(-1);
  }
}
