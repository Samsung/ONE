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

#include <ONNXImporterImpl.h>
#include <mir2loco.h>
#include <exo/TFLExporter.h>

#include <iostream>

namespace
{

// String decorators?
std::string quote(const std::string &s) { return "'" + s + "'"; }

std::unique_ptr<mir::Graph> import(const std::string &onnx_path)
{
  return mir_onnx::loadModel(onnx_path);
}

std::unique_ptr<loco::Graph> transform(const std::unique_ptr<mir::Graph> &mir_graph)
{
  mir2loco::Transformer transformer;
  return transformer.transform(mir_graph.get());
}

void printHelp()
{
  std::cout << "Usage: onnx2tflite <mode> <path/to/onnx> <path/to/output>\n"
               "Modes: -t (text file); -b (binary file)"
            << std::endl;
}

} // namespace

// ONNX-to-MIR (mir-onnx-importer)
// MIR-to-LOCO (mir2loco)
// LOCO-to-TFLITE (exo-tflite)
int main(int argc, char **argv)
{
  // onnx2tflite <mode> <path/to/onnx> <path/to/output>
  // modes: -t (text file); -b (binary file)
  if (argc != 4)
  {
    printHelp();
    exit(1);
  }
  std::string mode{argv[1]};
  std::string onnx_path{argv[2]};
  std::string tflite_path{argv[3]};

  std::cout << "Import " << quote(onnx_path) << std::endl;
  std::unique_ptr<mir::Graph> mir_graph;
  if (mode == "-t")
    mir_graph = mir_onnx::importModelFromTextFile(onnx_path);
  else if (mode == "-b")
    mir_graph = mir_onnx::importModelFromBinaryFile(onnx_path);
  else
  {
    printHelp();
    exit(1);
  }
  std::cout << "Import " << quote(onnx_path) << " - Done" << std::endl;

  auto loco_graph = transform(mir_graph);

  exo::TFLExporter(loco_graph.get()).dumpToFile(tflite_path.c_str());

  return 0;
}
