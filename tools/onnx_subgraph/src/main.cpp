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

#include "graph.h"
#include "device.h"
#include "partition.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
  std::string onnxFile;
  std::string confFile;

  if (argc == 3)
  {
    std::string arg1 = argv[1];
    if (arg1.substr(0, 7) == "--onnx=")
    {
      onnxFile = arg1.substr(7);
      if (onnxFile.empty())
      {
        std::cout << "No ONNX file provided." << std::endl;
        return -1;
      }

      if (!fs::exists(onnxFile))
      {
        std::cout << onnxFile << " not exists." << std::endl;
        return -1;
      }
      else
      {
        std::cout << onnxFile << " exists." << std::endl;
      }
    }

    std::string arg2 = argv[2];
    if (arg2.substr(0, 7) == "--conf=")
    {
      confFile = arg2.substr(7);
      if (confFile.empty())
      {
        std::cout << "No conf file provided." << std::endl;
        return -1;
      }

      if (!fs::exists(confFile))
      {
        std::cout << confFile << " not exists." << std::endl;
        return -1;
      }
      else
      {
        std::cout << confFile << " exists." << std::endl;
      }
    }
  }
  else
  {
    printf("Please set valide args: ./onnx-subgraph --onnx=xxx.onnx --conf=xxx.json\n");
    return -1;
  }

  auto g = GetGraphFromOnnx(onnxFile);

  Device target;
  target.updateOnnxFile(onnxFile);
  target.GetDeviceJson(confFile);
  std::unordered_map<std::string, NodeIOSize> node_io_size;
  PartitionGraph(g, target, PartitionStrategy::SPILTE_CPU_STRUCTURE_FIRST, node_io_size);
  std::cout << "PartitionGraph done." << std::endl;

  return 0;
}
