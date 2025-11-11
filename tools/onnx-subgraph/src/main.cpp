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

#include <iostream>
#include <string>
#include "graph.h"
#include "partition.h"
#include "Python.h"

int main(int argc, char *argv[])
{
  std::string onnxFile;
  if (argc > 1)
  {
    for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg.substr(0, 7) == "--onnx=")
      {
        onnxFile = arg.substr(7);
        std::cout << "ONNX file: " << onnxFile << std::endl;
      }
    }
    if (onnxFile.empty())
    {
      std::cout << "No ONNX file provided." << std::endl;
      return -1;
    }
  }
  else
  {
    printf("Please set valide args: ./onnx-subgraph --onnx=xxx.onnx\n");
    return -1;
  }

  Graph graph;
  auto g = graph.GetGraphFromOnnx(onnxFile);
  std::unordered_map<std::string, NodeIOSize> node_io_size;
  Partition p;
  Device target;
  target.updateOnnxFile(onnxFile);
  target.GetDeviceJson("./scripts/config.json");
  p.PartitionGraph(g, target, PartitionStrategy::SPILTE_NPU_STRUCTURE_FIRST, node_io_size);

  Py_Initialize();
  if (!Py_IsInitialized())
  {
    std::cout << "python init fail" << std::endl;
    return 0;
  }
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('.')");
  Py_Finalize();

  return 0;
}
