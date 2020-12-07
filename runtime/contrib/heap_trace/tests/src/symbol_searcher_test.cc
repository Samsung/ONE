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

#include "common_test_environment.h"
#include "file_content_manipulations.h"
#include "test_sample1.h"
#include "test_sample2.h"
#include "test_sample4.h"

#include "symbol_searcher.h"
#include "trace.h"

#include <dlfcn.h>
#include <linux/limits.h>
#include <unistd.h>

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

extern std::unique_ptr<Trace> GlobalTrace;

fs::path exePath()
{
  char result[PATH_MAX] = {0};
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  return fs::path(result).parent_path();
}

namespace backstage
{

struct SymbolSearcher : TestEnv
{
  SymbolSearcher() : TestEnv("./symbol_searcher_test.log") {}
};

TEST_F(SymbolSearcher, should_find_symbol_in_linked_library)
{
  ASSERT_TRUE((void *)funcDefinedOnlyInTestSample4 == findSymbol("funcDefinedOnlyInTestSample4"));
}

TEST_F(SymbolSearcher, should_find_symbol_in_library_which_have_been_loaded_in_runtime)
{
  fs::path pathToTestLib = exePath() / "libtest_sample2.so";
  void *handle = dlopen(pathToTestLib.c_str(), RTLD_NOW);

  ASSERT_TRUE(handle);
  ASSERT_TRUE(dlsym(handle, "funcDefinedOnlyInTestSample2") ==
              findSymbol("funcDefinedOnlyInTestSample2"));
  dlclose(handle);
}

TEST_F(SymbolSearcher,
       should_ignore_symbols_found_in_current_translation_unit_if_there_is_another_alternative)
{
  fs::path pathToTestSample2 = exePath() / "libtest_sample2.so";
  void *test_sample2_handle = dlopen(pathToTestSample2.c_str(), RTLD_NOW);
  void *func_addr_in_test_sample2 =
    dlsym(test_sample2_handle, "funcWhichCallFuncDefinedInTestSample3");

  ASSERT_TRUE(test_sample2_handle);
  ASSERT_TRUE((void *)funcDefinedInTestSample3_ButWrappedInTestSample1 !=
              reinterpret_cast<void *(*)()>(func_addr_in_test_sample2)());

  dlclose(test_sample2_handle);
}

TEST_F(SymbolSearcher, should_give_an_opportunity_do_not_log_its_internal_allocations)
{
  GlobalTrace.reset();
  fs::path pathToTestLib = exePath() / "libtest_sample2.so";
  void *handle = dlopen(pathToTestLib.c_str(), RTLD_NOW);

  GlobalTrace.reset(new Trace);
  void *symbolAddress = findSymbol("funcDefinedOnlyInTestSample2");
  GlobalTrace.reset();

  ASSERT_STREQ(getContentOfFile("./symbol_searcher_test.log").c_str(),
               "On CPU - Peak heap usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\nOn "
               "GPU - Peak mem usage: 0 B, Total allocated: 0 B, Total deallocated: 0 B\n");

  dlclose(handle);
}

} // namespace backstage
