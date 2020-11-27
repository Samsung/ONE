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

/*
 * This is simple tests to check that generator is running properly and creates compilable artifact
 * This test is not intended to check correctness of generated artifact
 */

#include "mir/Graph.h"
#include "mir/Shape.h"
#include "mir/ops/InputOp.h"
#include "mir/ops/OutputOp.h"
#include "mir/ops/ReluOp.h"

#include "backends/soft_backend/CPPGenerator.h"

#include <iostream>
#include <fstream>
#include <string>

#include <cstdlib>

// This header generated and contains array with test_main.def contents
#include "test_main.generated.h"

using namespace std;

using namespace nnc;
using namespace mir;

// Creates simple graph with input and output
static void fillGraph(Graph &g)
{
  Shape input_shape{1, 2, 3};
  mir::TensorType input_type{mir::DataType::FLOAT32, input_shape};
  Operation *input_op = g.create<ops::InputOp>(input_type);
  Operation *relu_op = g.create<ops::ReluOp>(input_op->getOutput(0));
  Operation *output_op = g.create<ops::OutputOp>(relu_op->getOutput(0));
  input_op->getOutput(0)->setName("in");
  relu_op->getOutput(0)->setName("out");
}

static void checkFileExists(const string &path)
{
  ifstream f(path);
  if (!f.good())
  {
    cerr << "file " << path << " not created\n";
    exit(1);
  }
}

static void createMain(const string &path, const string &header_path)
{
  ofstream out(path);
  if (!out.good())
  {
    cerr << "Main file " << path << " not created\n";
    exit(1);
  }
  out << "#include \"" << header_path << "\"\n";
  out.write(test_main, sizeof(test_main));
}

int main()
{
  std::string output_dir = "test_output";
  std::string artifact_name = "nnmodel";

  Graph g;
  fillGraph(g);

  nnc::CPPCodeGenerator cpp_code_generator(output_dir, artifact_name);
  cpp_code_generator.run(&g);

  string base_path = output_dir + "/" + artifact_name;

  string code_path = base_path + ".cpp";
  string header_path = base_path + ".h";
  string main_path = base_path + "_main.cpp";

  checkFileExists(code_path);
  checkFileExists(header_path);
  checkFileExists(base_path + ".params");

  createMain(main_path, artifact_name + ".h");

  string target_compiler = "g++ -Wall --std=c++11";

  string compiler_command =
    target_compiler + " -I" + output_dir + " " + main_path + " " + code_path;

  // call compiler
  int res = system(compiler_command.c_str());

  if (res == -1)
  {
    cerr << "failed to call compiler\n";
    return 2;
  }
  if (res != 0)
  {
    cerr << "compiler did not succeed with error code " << res << ": " << compiler_command << "\n";
    return 3;
  }
  return 0;
}
