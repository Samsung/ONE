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

#ifndef DEVICE_H
#define DEVICE_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "onnx.pb.h"
#include "graph.h"
#include <jsoncpp/json/json.h>

enum class DeviceType
{
  Target_NPU
};

class Device
{
private:
  std::string onnxFile;

public:
  Device(/* args */)
  {
    NPUPreferOp = {};
    CPUSupportOp = {};
    NPUSupportOp = {};
    max_subgraph_size = 0;
  }

  ~Device() {}

  std::vector<std::string> NPUPreferOp;
  std::vector<std::string> CPUSupportOp;
  std::vector<std::string> NPUSupportOp;

  float max_subgraph_size;

  DeviceType getType() { return DeviceType::Target_NPU; }

  std::vector<std::vector<std::string>> getCPUStructure()
  {
    return {{"Concat"},
            {"Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div"},
            {"Transpose", "Gather", "Gather", "Gather", "Transpose", "MatMul", "Mul", "Softmax",
             "MatMul"}};
  }

  std::vector<std::vector<std::string>> getNPUStructure()
  {
    return {{"Reshape", "Transpose", "Reshape"},
            {"Reshape", "Sigmoid", "Mul", "Transpose", "Conv", "Add", "Transpose"},
            {"Reshape", "Transpose", "Conv", "Transpose", "Reshape"},
            {"Reshape", "Conv", "Transpose"},
            {"Reshape", "Add", "Add", "Reshape", "Transpose", "Conv", "Add"},
            {"Conv"}};
  }

  std::vector<std::string> getNPUSupportOp() { return NPUSupportOp; }
  std::vector<std::string> getCPUSupportOp() { return CPUSupportOp; }
  std::vector<std::string> getNPUPreferOp() { return NPUPreferOp; }

  /**
   * @brief     Generate cut instructions for subgraphs based on the given device type.
   *
   * @param     [in] Subgraphs A reference to a vector of ONNX GraphProto objects representing
   * subgraphs.
   * @param     [in] device A string indicating the device type (e.g., "npu" or "c920").
   * @param     [in] subgraphs_inputs A reference to a vector of unordered sets containing input
   * information for subgraphs.
   * @param     [in] subgraphs_outputs A reference to a vector of unordered sets containing output
   * information for subgraphs.
   *
   * @pre       The function assumes that the `Subgraphs`, `subgraphs_inputs`, and
   * `subgraphs_outputs` vectors are properly initialized and have the same size.
   * @post      A file named `<device> CutInstruction.txt` is created or overwritten with the
   * generated cut instructions.
   * @exception If the output file cannot be opened, an error message is printed, and the program
   * exits.
   *
   * @return    None
   */
  void GenerateCutInstruction(std::vector<onnx::GraphProto> &Subgraphs, std::string device,
                              std::vector<std::unordered_set<NodeTensor>> &subgraphs_inputs,
                              std::vector<std::unordered_set<NodeTensor>> &subgraphs_outputs);

  /**
   * @brief Reads and parses a JSON file containing device information.
   *
   * This function reads a JSON file from the specified path, parses it, and extracts relevant
   * device information. It updates global variables with hardware limits, preferred NPU operations,
   * and supported operations for both NPU and CPU.
   *
   * @param json_path The file path to the JSON file containing device information.
   */
  void GetDeviceJson(std::string json_path)
  {
    Json::Reader reader;
    Json::Value root;

    // Open the JSON file in binary mode
    std::ifstream in(json_path, std::ios::binary);
    if (!in.is_open())
    {
      std::cout << "Error opening file\n";
      return;
    }

    if (reader.parse(in, root))
    {
      // Extract and set the maximum subgraph size from hardware limits
      float max_subgraph_size_json = root["hardware_limits"]["max_subgraph_size"].asFloat();
      max_subgraph_size = max_subgraph_size_json;
      // Iterate through performance data to identify operations where NPU outperforms CPU

      for (unsigned int i = 0; i < root["performance_data"].size(); i++)
      {
        if (root["performance_data"][i]["CPU_time"].asFloat() >
            root["performance_data"][i]["NPU_time"].asFloat())
        {
          NPUPreferOp.push_back(root["performance_data"][i]["name"].asString());
        }
      }

      // Iterate through and store supported NPU operations
      for (int i = 0; i < int(root["NPU_supported_ops"].size()); i++)
      {
        if (std::find(NPUSupportOp.begin(), NPUSupportOp.end(),
                      root["NPU_supported_ops"][i].asString()) == NPUSupportOp.end())
        {
          NPUSupportOp.push_back(root["NPU_supported_ops"][i].asString());
        }
      }

      // Iterate through and store supported CPU operations
      for (int i = 0; i < int(root["CPU_supported_ops"].size()); i++)
      {
        if (std::find(CPUSupportOp.begin(), CPUSupportOp.end(),
                      root["CPU_supported_ops"][i].asString()) == CPUSupportOp.end())
        {
          CPUSupportOp.push_back(root["CPU_supported_ops"][i].asString());
        }
      }
    }

    in.close();
  }

  void updateOnnxFile(std::string &path) { onnxFile = path; }

  std::string getOnnxFile() { return onnxFile; }
};

#endif
