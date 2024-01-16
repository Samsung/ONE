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

#include "backends/acl_soft_backend/AclCppGenerator.h"
#include "AclCppOpGenerator.h"
#include "backends/acl_soft_backend/AclCppException.h"

#include <filesystem>
#include <fstream>
#include <utility>

namespace nnc
{

using namespace std;
namespace fs = std::filesystem;

AclCppCodeGenerator::AclCppCodeGenerator(string output_dir, string artifact_name)
  : _output_dir(std::move(output_dir)), _artifact_name(std::move(artifact_name))
{
}

void AclCppCodeGenerator::run(mir::Graph *data)
{
  mir::Graph *g = data;
  assert(g);

  // Create a directory for generated artifact files.
  fs::create_directory(_output_dir);

  string base_path = _output_dir + "/" + _artifact_name;
  string code_path = base_path + ".cpp";
  string decl_path = base_path + ".h";
  string par_path = base_path + ".par";

  // Create the source and header files output streams.
  ofstream code_out(code_path);

  if (code_out.fail())
    throw AclCppException("Can not open code output file: " + code_path);

  ofstream decl_out(decl_path);

  if (decl_out.fail())
    throw AclCppException("Can not open declaration output file: " + decl_path);

  ofstream par_out(par_path, ios_base::out | ios_base::binary);

  if (par_out.fail())
    throw AclCppException("Can not open parameter output file: " + par_path);

  ArtifactGeneratorCppCode code_gen(code_out);
  ArtifactGeneratorCppDecl decl_gen(decl_out);

  // Generate the artifact.
  AclCppOpGenerator op_generator(_artifact_name, par_out);
  auto dom = op_generator.generate(g);
  dom.accept(&code_gen);
  dom.accept(&decl_gen);
}

} // namespace nnc
