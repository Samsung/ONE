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
    exit(0);
  }
  for (size_t i = 0; i < Subgraphs.size(); i++)
  {
    // default parameters
    std::string modelFile = onnxFile;
    std::string dataScaleDiv = "255";
    std::string postprocess = "save_and_top5";

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

    std::string calibrateDataset = device + "_Subgraphs_" + std::to_string(i) + ".npz";
    std::string quantizationScheme = "int8_asym";
  }

  outFile.close();
}
