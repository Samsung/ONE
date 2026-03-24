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

#ifndef _NNC_SOFT_BACKEND_CPP_GENERATOR_H_
#define _NNC_SOFT_BACKEND_CPP_GENERATOR_H_

#include "mir/Graph.h"

#include <ostream>
#include <string>
#include <vector>

namespace nnc
{

class ModelAnalyzer;
class Serializer;

namespace sir
{
struct TensorDescriptor;
struct Action;
struct CallFunction;
struct TransposeTensor;
struct CreateTmp;
struct DestroyTmp;
} // namespace sir

/**
 * @brief CPPCodeGenerator implements interfaces that provides BaseCodeGenerator for C++ language
 * This includes header file generation, code file generation and variable renaming according to C++
 * naming requirements
 */
class CPPCodeGenerator final
{
public:
  CPPCodeGenerator(std::string output_dir, std::string artifact_name);

  /**
   * @brief Method represents base generation sequence: analysis, serialization, header/code
   * generation, etc
   * @param graph MIR graph
   */
  void run(mir::Graph *graph);

private:
  /**
   * @brief This function processes tensor names
   * to transform them into valid identificators of target language
   * @param ma Intermediate artifact information
   */
  void formatTensorNames(const ModelAnalyzer &ma);
  /**
   * @brief Derivative classes should override this function to generate header of artifact
   * @param out Stream to write header text
   * @param ma Intermediate artifact information
   */
  void materializeHeader(std::ostream &out, const ModelAnalyzer &ma);

  /**
   * @brief Form list of function call arguments
   * @param ma Intermediate model representation
   * @param argIds List of argument variable ids
   * @param args Result list of arguments transformed in form of strings
   */
  void gatherOperationArguments(const ModelAnalyzer &ma, const std::vector<std::size_t> &arg_ids,
                                std::vector<std::string> &args);
  /**
   * @brief Prints setter of artifact
   * @param out Output stream
   * @param className Name of artifact
   * @param setterName Name of setter function
   * @param varId id of variable that setter fills
   */
  void printSetter(std::ostream &out, const std::string &class_name, const std::string &setter_name,
                   const sir::TensorDescriptor &td);
  /**
   * @brief Prints getters of artifact
   * @param out Output stream
   * @param className Name of artifact
   * @param setterName Name of setter function
   * @param varId id of variable that getter returns
   */
  void printGetter(std::ostream &out, const std::string &class_name, const std::string &getter_name,
                   const sir::TensorDescriptor &td);
  /**
   * @brief Generate code for function call action
   * @param out Output stream to print
   * @param ma Intermediate model representation
   * @param call Action to generate code from
   */
  void materializeCall(std::ostream &out, const ModelAnalyzer &ma, const sir::CallFunction *call);
  /**
   * @brief Generate code for transpose action
   * @param out Output stream to print
   * @param ma Intermediate model representation
   * @param action Action to generate code from
   */
  void materializeTranspose(std::ostream &out, const ModelAnalyzer &ma,
                            const sir::TransposeTensor *transpose);
  /**
   * @brief Generate code for constructor action
   * @param out Output stream to print
   * @param ma Intermediate model representation
   * @param action Action to generate code from
   */
  void materializeConstructor(std::ostream &out, const ModelAnalyzer &ma,
                              const sir::CreateTmp *constructor);
  /**
   * @brief Generate code for destructor action
   * @param out Output stream to print
   * @param ma Intermediate model representation
   * @param action Action to generate code from
   */
  void materializeDestructor(std::ostream &out, const ModelAnalyzer &ma,
                             const sir::DestroyTmp *destructor);
  /**
   * @brief Prints inference sequence placed in doInference method of artifact
   * @param out Output stream
   * @param ma Intermediate model representation
   */
  void materializeInferenceSequence(std::ostream &out, const ModelAnalyzer &ma);
  /**
   * @brief Derivative classes should override this function to generate implementation of artifact
   * @param out Stream to write header text
   * @param ma Intermediate artifact information
   * @param s Serializer holds parameters of network and various meta-information: serializer
   * version, hashes, etc
   */
  void materializeCode(std::ostream &out, const ModelAnalyzer &ma, const Serializer &s);
  /**
   * @brief Writes serialized parameters to out stream
   * @param out Stream to write serialized parameters
   * @param s Serializer holds parameters of network
   *
   * Contents of generated file:
   * + header(magic number to identify file type, protocol version, hashes of network and params)
   * + array of serialized network parameters
   */
  void materializeModelParams(std::ostream &out, const Serializer &s);

  std::string _output_dir;
  std::string _artifact_name;
  std::vector<std::string> _formattedTensors;
};

} // namespace nnc

#endif //_NNC_SOFT_BACKEND_CPP_GENERATOR_H_
