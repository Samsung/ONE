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

#include "allocation.h"
#include "args.h"
#include "ggma.h"
#include "ggma_internal.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(const int argc, char **argv)
{
  using namespace ggma_run;

  try
  {
    Args args(argc, argv);

    ggma_pkg *pkg = nullptr;
    GGMA_ENSURE(ggma_new_package(&pkg, args.packagePath().c_str()));

    std::string prompt = "Lily picked up a flower.";
    constexpr size_t n_tokens_max = 32;
    ggma_token tokens[n_tokens_max];
    size_t n_tokens;
    GGMA_ENSURE(ggma_tokenize(pkg, prompt.c_str(), prompt.size(), tokens, n_tokens_max, &n_tokens));

    ggma_session *session = nullptr;
    GGMA_ENSURE(ggma_create_session_with_package(&session, pkg));
    GGMA_ENSURE(ggma_set_config(session, "ENABLE_LOG", std::getenv("ENABLE_LOG")));
    GGMA_ENSURE(ggma_set_config(session, "NUM_THREADS", std::getenv("NUM_THREADS")));

    size_t n_tokens_out;
    GGMA_ENSURE(ggma_generate(session, tokens, n_tokens, n_tokens_max, &n_tokens_out));
    GGMA_ENSURE(ggma_close_session(session));
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error: " << e.what() << std::endl;
    exit(-1);
  }
}
