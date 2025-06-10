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

#ifndef __TOOLS_ONNX_SUBGRAPH_DEVICE_H__
#define __TOOLS_ONNX_SUBGRAPH_DEVICE_H__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <jsoncpp/json/json.h>
#include "onnx.pb.h"
#include "graph.h"

enum class DeviceType
{
  Target_NPU
};

class Device
{
private:
  std::string _onnxFile;

public:
  Device(/* args */)
  {
    _NPUPreferOp = {};
    _CPUSupportOp = {};
    _NPUSupportOp = {};
    _max_subgraph_size = 0;
  }

  std::vector<std::string> _NPUPreferOp;
  std::vector<std::string> _CPUSupportOp;
  std::vector<std::string> _NPUSupportOp;
  float _max_subgraph_size;

  DeviceType getType() { return DeviceType::Target_NPU; }
  std::vector<std::string> getNPUSupportOp() { return _NPUSupportOp; }
  std::vector<std::string> getCPUSupportOp() { return _CPUSupportOp; }
  std::vector<std::string> getNPUPreferOp() { return _NPUPreferOp; }

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
  void GetDeviceJson(const std::string &json_path);

  void updateOnnxFile(std::string &path) { _onnxFile = path; }

  std::string getOnnxFile() { return _onnxFile; }
};

#endif //__TOOLS_ONNX_SUBGRAPH_DEVICE_H__
