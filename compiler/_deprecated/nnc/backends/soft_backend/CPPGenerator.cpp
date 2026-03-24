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

#include "backends/soft_backend/CPPGenerator.h"

#include "mir/Operation.h"
#include "ModelAnalyzer.h"
#include "SBSerializer.h"

#include "CommonData.def"

#include "cpp_header_types.generated.h"
#include "cpp_operations.generated.h"
#include "CommonData.generated.h"
#include "eigen.generated.h"
#include "cpp_common_funcs.generated.h"
#include "cpp_capped_relu.generated.h"
#include "cpp_concat.generated.h"
#include "cpp_conv.generated.h"
#include "cpp_conv_transpose.generated.h"
#include "cpp_depthwise_conv.generated.h"
#include "cpp_fully_connected.generated.h"
#include "cpp_pool.generated.h"
#include "cpp_sigmoid.generated.h"
#include "cpp_sqrt.generated.h"
#include "cpp_relu.generated.h"
#include "cpp_leaky_relu.generated.h"
#include "cpp_reduce.generated.h"
#include "cpp_resize.generated.h"
#include "cpp_softmax.generated.h"
#include "cpp_slice.generated.h"
#include "cpp_elu.generated.h"
#include "cpp_tanh.generated.h"
#include "cpp_elementwise.generated.h"
#include "cpp_pad.generated.h"
#include "cpp_transpose.generated.h"
#include "cpp_gather.generated.h"
#include "cpp_broadcast.generated.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace nnc
{

using namespace sir;
using namespace std;
namespace fs = std::filesystem;

/**
 * @brief Creates pointer to some output stream to encapsulate resource management into deleter
 * for example may be used to return std::cout
 * @param path Path to opened file
 * @return Pointer output stream
 * @throws runtime_error if did not succeed
 */
static unique_ptr<ofstream> getStream(const string &path)
{
  unique_ptr<ofstream> ofs(new ofstream(path));
  if (ofs->fail())
    throw runtime_error("Can not open code output file: " + path);
  return ofs;
}

CPPCodeGenerator::CPPCodeGenerator(std::string output_dir, std::string artifact_name)
  : _output_dir(std::move(output_dir)), _artifact_name(std::move(artifact_name))
{
}

void CPPCodeGenerator::materializeModelParams(ostream &out, const Serializer &s)
{
  using namespace params;

  // First form a dump header
  char header[HEADER_LEN];
  uint32_t version = s.getFormatVersion();
  uint32_t hash = s.getModelHash();
  static_assert(VERSION_LEN == sizeof(version), "version length mismatch");
  static_assert(HASH_LEN == sizeof(hash), "hash length mismatch");
  memcpy(header, MAGIC, MAGIC_LEN);
  memcpy(header + MAGIC_LEN, &version, VERSION_LEN);
  memcpy(header + MAGIC_LEN + VERSION_LEN, &hash, HASH_LEN);

  out.write(header, HEADER_LEN);
  if (out.fail())
    throw runtime_error("Failed to write model parameters header");
  auto &params = s.getBuffer();
  out.write(params.data(), params.size());
  if (out.fail())
    throw runtime_error("Failed to write model Parameters");
}

void CPPCodeGenerator::run(mir::Graph *graph)
{
  assert(graph);

  // visit and analyze graph
  ModelAnalyzer ma;
  ma.analyze(graph);
  // serialize parameters
  Serializer serializer;
  serializer.serialize(ma.getInferenceSequence());
  // rename tensors for specific backend language
  formatTensorNames(ma);

  fs::create_directory(_output_dir);

  const string base_path = _output_dir + "/" + _artifact_name;
  const string header_path = base_path + ".h";
  const string code_path = base_path + ".cpp";
  const string params_path = base_path + ".params";

  // Print header
  auto header_stream = getStream(header_path);
  materializeHeader(*header_stream, ma);
  header_stream.reset();

  // Print code
  auto code_stream = getStream(code_path);
  materializeCode(*code_stream, ma, serializer);
  code_stream.reset();

  // Print model parameters
  auto model_stream = getStream(params_path);
  materializeModelParams(*model_stream, serializer);
  model_stream.reset();
}

/**
 * @brief Renames tensors with respect to C++ naming conventions
 * @param ma Intermediate artifact information
 */
void CPPCodeGenerator::formatTensorNames(const ModelAnalyzer &ma)
{
  using TensorType = TensorDescriptor::Type;

  int tmp_tensors = 0;
  for (const TensorDescriptor &td : ma.getTensors())
  {
    string formatted_name;
    if (td.name.empty())
    {
      assert(td.type == TensorType::temporary);
      formatted_name = "Tensor_" + to_string(tmp_tensors++);
    }
    else
    {
      if (td.type != TensorType::temporary)
        formatted_name.append("_");
      formatted_name.append(td.name);
      for (char &c : formatted_name)
      {
        if (!isalnum(c))
          c = '_';
      }
    }
    _formattedTensors.push_back(move(formatted_name));
  }
}

/**
 * + Writes to out support data types and methods: Shape, Tensor.
 * This is part of user interface to feed data to artifact.
 * + Writes actual model class that contains:
 * network constructor, setters to feed data to network, getters to get results,
 * and doInference method that performs actual inference.
 */
void CPPCodeGenerator::materializeHeader(ostream &out, const ModelAnalyzer &ma)
{
  string class_name = ma.getModelName() + "Model";

  out.write(cpp_header_types, sizeof(cpp_header_types));
  out << "class " << class_name
      << "\n"
         "{\n"
         "public:\n"
         "  "
      << class_name
      << "(const std::string& parametersPath);\n"
         "  ~"
      << class_name << "();\n";
  // generate input setters
  if (ma.getInputs().size() == 1)
    out << "  bool setInput(const Tensor& inputs);\n";
  for (const size_t inId : ma.getInputs())
  {
    const string &tName = _formattedTensors[inId];
    out << "  bool set" << tName << "(const Tensor& t);\n";
  }
  // generate output getters
  if (ma.getOutputs().size() == 1)
  {
    out << "  std::shared_ptr<Tensor> getOutput();\n";
  }
  for (const size_t out_id : ma.getPersistentTensors())
  {
    const string &tensor_name = _formattedTensors[out_id];
    out << "  std::shared_ptr<Tensor> get" << tensor_name << "();\n";
  }
  out << "  void doInference();\n\n"
         "private:\n"
         "  "
      << class_name
      << "() = delete;\n"
         "  "
      << class_name << "(const " << class_name
      << "& orig) = delete;\n"
         "  "
      << class_name << "& operator=(const " << class_name << "& orig) = delete;\n";
  // generate input/output tensors
  for (const size_t in_tensor_id : ma.getInputs())
  {
    const string &tName = _formattedTensors[in_tensor_id];
    out << "  Tensor " << tName << ";\n";
  }
  for (const size_t out_tensor_id : ma.getPersistentTensors())
  {
    const string &tName = _formattedTensors[out_tensor_id];
    out << "  std::shared_ptr<Tensor> " << tName << ";\n";
  }
  // pointer to NN parameters
  out << "  char* _parameters;\n";
  out << "  size_t _paramSize;\n";
  out << "};\n";
}

/**
 * @brief Prints list of function arguments, separated by commas
 * @param out Stream to write program text
 * @param args arguments to print
 */
static void printOperationArgs(ostream &out, const vector<string> &args)
{
  bool insert_comma = false;
  for (const string &arg : args)
  {
    if (insert_comma)
      out << ", ";
    insert_comma = true;
    out << arg;
  }
}

void CPPCodeGenerator::gatherOperationArguments(const ModelAnalyzer &ma,
                                                const vector<size_t> &arg_ids, vector<string> &args)
{

  for (size_t id : arg_ids)
  {
    const string &tensor_name = _formattedTensors[id];
    if (ma.getTensors()[id].type == TensorDescriptor::Type::persistent)
      args.push_back("*" + tensor_name);
    else
      args.push_back(tensor_name);
  }
}

void CPPCodeGenerator::printSetter(ostream &out, const string &class_name,
                                   const string &setter_name, const TensorDescriptor &td)
{

  const string &var_name = _formattedTensors[td.id];
  out << "bool " << class_name << "::set" << setter_name
      << "(const Tensor& t)\n"
         "{\n";
  // need to insert input correctness check
  const mir::Shape expected = td.shape;
  int rank = expected.rank();
  if (rank != 0)
  {
    out << "  "
        << "if (t.getShape().getDims() != " << td.shape.rank() << ") return false;\n";
    for (int i = 0; i < rank; ++i)
      out << "  "
          << "if (t.getShape()[" << i << "] != " << expected.dim(i) << ") return false;\n";
  }
  out << "  " << var_name
      << " = t;\n"
         "  return true;\n"
         "}\n\n";
}

void CPPCodeGenerator::printGetter(ostream &out, const string &class_name,
                                   const string &getter_name, const TensorDescriptor &td)
{

  const string &var_name = _formattedTensors[td.id];
  out << "shared_ptr<Tensor> " << class_name << "::get" << getter_name
      << "()\n"
         "{\n"
         "  return "
      << var_name
      << ";\n"
         "}\n\n";
}

void CPPCodeGenerator::materializeCall(ostream &out, const ModelAnalyzer &ma,
                                       const sir::CallFunction *call)
{
  assert(call != nullptr);
  if (call->mirOp->getType() == mir::Operation::Type::input)
    return;
  // materialize call
  out << "  " << call->funcName << "(";
  const auto &prev_nodes = call->mirOp->getInputs();
  const auto &out_tensors = call->outputs;
  vector<string> args;
  args.reserve(prev_nodes.size() + out_tensors.size() + 1);
  // gather output arguments
  gatherOperationArguments(ma, call->outputs, args);
  // parameters offset
  args.push_back("_parameters + " + to_string(params::HEADER_LEN + call->paramStartOffset));
  // gather input arguments
  gatherOperationArguments(ma, call->inputs, args);
  // put arguments into stream
  printOperationArgs(out, args);
  out << ");\n";
}

void CPPCodeGenerator::materializeTranspose(ostream &out, const ModelAnalyzer &ma,
                                            const sir::TransposeTensor *transpose)
{
  assert(transpose != nullptr);
  (void)out;
  (void)ma;
  (void)transpose;
  assert(false && "not implemented");
}

void CPPCodeGenerator::materializeConstructor(ostream &out, const ModelAnalyzer &ma,
                                              const sir::CreateTmp *constructor)
{
  assert(constructor != nullptr);
  const TensorDescriptor &td = ma.getTensors()[constructor->tensorId];
  assert(td.type == sir::TensorDescriptor::Type::temporary);
  (void)td;
  const string &t_name = _formattedTensors[constructor->tensorId];
  out << "  Tensor " << t_name << ";\n";
}

void CPPCodeGenerator::materializeDestructor(ostream &out, const ModelAnalyzer &ma,
                                             const sir::DestroyTmp *destructor)
{
  assert(destructor != nullptr);
  const TensorDescriptor &td = ma.getTensors()[destructor->tensorId];
  assert(td.type == sir::TensorDescriptor::Type::temporary);
  (void)td;
  const string &t_name = _formattedTensors[destructor->tensorId];
  out << "  " << t_name << ".clean();\n";
}

void CPPCodeGenerator::materializeInferenceSequence(ostream &out, const ModelAnalyzer &ma)
{

  // Allocate temporary(im2col) tensor
  out << "  Tensor " << _formattedTensors[ma.getTempTID()] << "(Shape{" << ma.getMaxTemporarySize()
      << "});\n";

  for (const unique_ptr<Action> &action : ma.getInferenceSequence())
  {
    Action *ptr = action.get();
    switch (action->type)
    {
      case Action::Type::callFunction:
        materializeCall(out, ma, dynamic_cast<const sir::CallFunction *>(ptr));
        break;
      case Action::Type::transposeTensor:
        materializeTranspose(out, ma, dynamic_cast<const sir::TransposeTensor *>(ptr));
        break;
      case Action::Type::createTmp:
        materializeConstructor(out, ma, dynamic_cast<const sir::CreateTmp *>(ptr));
        break;
      case Action::Type::destroyTmp:
        materializeDestructor(out, ma, dynamic_cast<const sir::DestroyTmp *>(ptr));
        break;
      default:
        assert(false && "unexpected action type");
    }
  }
}

/**
 * Function writes to output stream needed code snippets, and implementations of artifact class
 * functions.
 */
void CPPCodeGenerator::materializeCode(ostream &out, const ModelAnalyzer &ma, const Serializer &s)
{
  string class_name = ma.getModelName() + "Model";

  out << "#include \"" << _artifact_name << ".h\"\n";

  // put operations from tflite
  out.write(eigen, sizeof(eigen));

  out.write(CommonData, sizeof(CommonData));

  out.write(cpp_common_funcs, sizeof(cpp_common_funcs));
  out.write(cpp_capped_relu, sizeof(cpp_capped_relu));
  out.write(cpp_concat, sizeof(cpp_concat));
  out.write(cpp_conv, sizeof(cpp_conv));
  out.write(cpp_depthwise_conv, sizeof(cpp_depthwise_conv));
  out.write(cpp_fully_connected, sizeof(cpp_fully_connected));
  out.write(cpp_resize, sizeof(cpp_resize));
  out.write(cpp_sigmoid, sizeof(cpp_sigmoid));
  out.write(cpp_pool, sizeof(cpp_pool));
  out.write(cpp_relu, sizeof(cpp_relu));
  out.write(cpp_reduce, sizeof(cpp_reduce));
  out.write(cpp_softmax, sizeof(cpp_softmax));
  out.write(cpp_slice, sizeof(cpp_slice));
  out.write(cpp_elementwise, sizeof(cpp_elementwise));
  out.write(cpp_elu, sizeof(cpp_elu));
  out.write(cpp_tanh, sizeof(cpp_tanh));
  out.write(cpp_pad, sizeof(cpp_pad));
  out.write(cpp_sqrt, sizeof(cpp_sqrt));
  out.write(cpp_conv_transpose, sizeof(cpp_conv_transpose));
  out.write(cpp_transpose, sizeof(cpp_transpose));
  out.write(cpp_gather, sizeof(cpp_gather));
  out.write(cpp_broadcast, sizeof(cpp_broadcast));
  // Operations calls into all of the above
  out.write(cpp_operations, sizeof(cpp_operations));
  // Below call into operations
  out.write(cpp_leaky_relu, sizeof(cpp_leaky_relu));

  // gen NN constructor
  out << class_name << "::" << class_name
      << "(const string& parametersPath)\n"
         "{\n"
         "  readParameters(_parameters, _paramSize, parametersPath, "
      << s.getFormatVersion() << ", " << s.getModelHash()
      << ");\n"
         "}\n\n";
  // gen NN destructor
  out << class_name << "::~" << class_name
      << "()\n"
         "{\n"
         "  releaseParameters(_parameters, _paramSize);\n"
         "}\n\n";
  // generate input setters
  // generate main setter if network has only one
  const auto &inputs = ma.getInputs();
  const auto &tensors = ma.getTensors();
  if (inputs.size() == 1)
  {
    const TensorDescriptor &td = tensors[inputs[0]];
    printSetter(out, class_name, "Input", td);
  }
  // generate setters by names
  for (size_t input_tensor_id : inputs)
  {
    const string &input_tensor_name = _formattedTensors[input_tensor_id];
    const TensorDescriptor &td = tensors[input_tensor_id];
    printSetter(out, class_name, input_tensor_name, td);
  }

  // gen output getters
  // generate main getter if network has only one
  const auto &outputs = ma.getOutputs();
  if (outputs.size() == 1)
  {
    const TensorDescriptor &td = tensors[outputs[0]];
    printGetter(out, class_name, "Output", td);
  }
  for (size_t output_tensor_id : ma.getPersistentTensors())
  {
    const string &output_tensor_name = _formattedTensors[output_tensor_id];
    const TensorDescriptor &td = tensors[output_tensor_id];
    printGetter(out, class_name, output_tensor_name, td);
  }
  out << "void " << class_name
      << "::doInference()\n"
         "{\n";
  for (size_t output_tensor_id : ma.getPersistentTensors())
  {
    const string &output_tensor_name = _formattedTensors[output_tensor_id];
    out << "  " << output_tensor_name << ".reset(new Tensor());\n";
  }

  // gen inference sequence
  materializeInferenceSequence(out, ma);
  out << "}";
}

} // namespace nnc
