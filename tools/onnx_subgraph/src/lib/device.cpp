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
#include "device.h"

void Device::GetDeviceJson(const std::string &json_path)
{
  Json::Reader reader;
  Json::Value root;

  // Open the JSON file in binary mode
  std::ifstream in(json_path, std::ios::binary);
  if (!in.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(-1);
  }

  if (reader.parse(in, root))
  {
    // Extract and set the maximum subgraph size from hardware limits
    float max_subgraph_size_json = root["hardware_limits"]["max_subgraph_size"].asFloat();
    _max_subgraph_size = max_subgraph_size_json;

    // Iterate through performance data to identify operations where NPU outperforms CPU
    for (unsigned int i = 0; i < root["performance_data"].size(); i++)
    {
      if (root["performance_data"][i]["CPU_time"].asFloat() >
          root["performance_data"][i]["NPU_time"].asFloat())
      {
        _NPUPreferOp.push_back(root["performance_data"][i]["name"].asString());
        std::cout << "Performance Op: " << root["performance_data"][i]["name"].asString()
                  << std::endl;
      }
    }

    // Iterate through and store supported NPU operations
    for (int i = 0; i < int(root["NPU_supported_ops"].size()); i++)
    {
      if (std::find(_NPUSupportOp.begin(), _NPUSupportOp.end(),
                    root["NPU_supported_ops"][i].asString()) == _NPUSupportOp.end())
      {
        _NPUSupportOp.push_back(root["NPU_supported_ops"][i].asString());
        std::cout << "NPU Supported: " << root["NPU_supported_ops"][i].asString() << std::endl;
      }
    }

    // Iterate through and store supported CPU operations
    for (int i = 0; i < int(root["CPU_supported_ops"].size()); i++)
    {
      if (std::find(_CPUSupportOp.begin(), _CPUSupportOp.end(),
                    root["CPU_supported_ops"][i].asString()) == _CPUSupportOp.end())
      {
        _CPUSupportOp.push_back(root["CPU_supported_ops"][i].asString());
        std::cout << "CPU Supported: " << root["CPU_supported_ops"][i].asString() << std::endl;
      }
    }
  }
}

void Device::GenerateCutInstruction(std::vector<onnx::GraphProto> &Subgraphs, std::string device,
                                    std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs,
                                    std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs)
{
  std::cout << "Generate Cut Instruction for Target_NPU" << std::endl;
  // open file
  std::string file_name = device + "CutInstruction.txt";
  std::ofstream outFile(file_name);

  if (!outFile.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(-1);
  }

  for (size_t i = 0; i < Subgraphs.size(); i++)
  {
    // default parameters
    std::string modelFile = _onnxFile;

    std::unordered_set<NodeTensor> graphInputs = subgraphs_inputs[i];
    std::unordered_set<NodeTensor> graphOutputs = subgraphs_outputs[i];

    std::string inputName = "\"";
    for (const auto &input : graphInputs)
    {
      inputName = inputName + input.name + ";";
    }

    // delete last semicolon
    if (!inputName.empty() && inputName.back() == ';')
    {
      inputName.pop_back();
    }

    inputName = inputName + "\"";
    std::string outputName = "\"";

    for (const auto &output : graphOutputs)
    {
      outputName = outputName + output.name + ";";
    }

    // delete last semicolon
    if (!outputName.empty() && outputName.back() == ';')
    {
      outputName.pop_back();
    }
    outputName = outputName + "\"";

    std::string inputShape = "\"";
    for (const auto &input : graphInputs)
    {
      for (const auto &dim : input.shape)
      {
        inputShape = inputShape + std::to_string(dim) + " ";
      }

      // delete last space
      if (!inputShape.empty() && inputShape.back() == ' ')
      {
        inputShape.pop_back();
      }
      inputShape = inputShape + ";";
    }

    // delete last semicolon
    if (!inputShape.empty() && inputShape.back() == ';')
    {
      inputShape.pop_back();
    }
    inputShape = inputShape + "\"";
  }

  outFile.close();
}
