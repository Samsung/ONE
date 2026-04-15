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

#ifndef NNCC_COMMANDLINEARGUMENTS_H
#define NNCC_COMMANDLINEARGUMENTS_H

#include <string>
#include "support/CommandLine.h"

namespace nnc
{
namespace cli
{

/**
 * Options for compiler driver
 */
extern Option<bool> caffe2Frontend; // frontend for CAFFE2 AI framework
extern Option<std::vector<int>> inputShapes;
extern Option<std::string> initNet;

extern Option<bool> caffeFrontend; // frontend for CAFFE AI framework
extern Option<bool> tflFrontend;   // frontend for TensorFlow Lite AI framework
extern Option<bool> onnxFrontend;  // frontend for ONNX AI framework

extern Option<bool> doOptimizationPass; // enable optimization pass
extern Option<bool> dumpGraph;          // enable Dumping graph to .dot files

// valid values for target option
#define NNC_TARGET_ARM_CPP "arm-c++"
#define NNC_TARGET_X86_CPP "x86-c++"
#define NNC_TARGET_ARM_GPU_CPP "arm-gpu-c++"
#define NNC_TARGET_INTERPRETER "interpreter"
extern Option<std::string> target; // kind of target for which compiler generates code

/**
 * Frontend options
 */
extern Option<std::string> inputFile; // files contains model of specific AI framework

/**
 * Options for backend
 */
extern Option<std::string> artifactDir;  // output directory for artifact
extern Option<std::string> artifactName; // name of artifact

/**
 * Options for interpreter
 */
extern Option<std::string> interInputDataDir; // directory with input data files

} // namespace cli
} // namespace nnc

#endif // NNCC_COMMANDLINEARGUMENTS_H
