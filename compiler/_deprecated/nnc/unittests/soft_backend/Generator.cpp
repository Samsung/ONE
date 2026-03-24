/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "backends/soft_backend/CPPGenerator.h"
#include "mir/ops/ReluOp.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <ftw.h>

using namespace std;

using namespace nnc;
using namespace mir;

static bool isFileExists(const string &path)
{
  ifstream f(path);
  return f.good();
}

static void deleteFile(const string &path)
{
  int res = remove(path.c_str());
  assert(!res && "failed to remove file");
  (void)res;
}

int removeRec(const char *fpath, const struct stat * /*sb*/, int /*typeflag*/,
              struct FTW * /*ftwbuf*/)
{
  deleteFile(fpath);
  return 0;
}

static void deleteDir(const string &path)
{
  int res = nftw(path.c_str(), removeRec, 1, FTW_DEPTH | FTW_PHYS);
  assert(!res && "failed to remove dir");
  (void)res;
}

static void checkOutputExists(const string &common_path)
{
  ASSERT_TRUE(isFileExists(common_path + ".h"));
  ASSERT_TRUE(isFileExists(common_path + ".cpp"));
  ASSERT_TRUE(isFileExists(common_path + ".params"));
}

static void emptyFile(const string &path) { ofstream of(path); }

TEST(Generator, check_generator_call)
{
// assume here that c++ and c code generators behave identically in terms of parameters check
// test only c++ generator
#define TEST_DIR "output_dir"
#define TEST_NAME "someName"
#define BASE_NAME TEST_DIR "/" TEST_NAME

  mir::Graph g;
  mir::TensorType input_type{mir::DataType::FLOAT32, Shape{1, 2, 3, 4}};
  Operation::Output *input = g.create<ops::InputOp>(input_type)->getOutput(0);
  input->setName("input");
  Operation *output = g.create<ops::ReluOp>(input);

  // test that generator creates output dir and files
  if (isFileExists(TEST_DIR))
    deleteDir(TEST_DIR);
  assert(!isFileExists(TEST_DIR) && "remove output dir");
  CPPCodeGenerator cpp_code_generator(TEST_DIR, TEST_NAME);
  cpp_code_generator.run(&g);
  checkOutputExists(BASE_NAME);

  // test that generator creates output files in existing empty dir
  deleteFile(BASE_NAME ".h");
  deleteFile(BASE_NAME ".cpp");
  deleteFile(BASE_NAME ".params");
  cpp_code_generator.run(&g);
  checkOutputExists(BASE_NAME);

  // test that generator rewrites existing files
  emptyFile(BASE_NAME ".h");
  struct stat sBefore, sAfter;
  int res = stat(BASE_NAME ".h", &sBefore);
  assert(res == 0);
  (void)res;
  assert(sBefore.st_size == 0);
  cpp_code_generator.run(&g);
  res = stat(BASE_NAME ".h", &sAfter);
  assert(res == 0);

  ASSERT_NE(sBefore.st_size, sAfter.st_size);

  deleteDir(TEST_DIR);
}
