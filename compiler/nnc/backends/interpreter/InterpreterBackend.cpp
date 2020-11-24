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

#include <cstring>
#include <utility>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "Definitions.h"

#ifdef NNC_HDF5_SUPPORTED
#include <H5Cpp.h>
#else
#include <iostream>
#endif // NNC_HDF5_SUPPORTED

#include "mir/Shape.h"

#include "MirInterpreter.h"
#include "backends/interpreter/InterpreterBackend.h"

#include "mir/Graph.h"

#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include <stdexcept>

namespace nnc
{

using namespace mir;

#ifdef NNC_HDF5_SUPPORTED

/**
 * @brief save tensor in file in '.hdf5' format
 * @param tensor - tensor to save
 * @param tensor_name - name, by wich tensor will be saved
 * @param destination - path to file, in which tensor will be saved
 */
static void writeTensorToHDF5File(const TensorVariant &tensor, std::string tensor_name,
                                  const std::string &destination)
{

  // Prepare shape, rank, dims, numElems
  auto &shape = tensor.getShape();
  const int32_t rank = shape.rank();
  hsize_t dims[rank];
  for (int32_t axis = 0; axis < rank; ++axis)
  {
    dims[axis] = static_cast<hsize_t>(shape.dim(axis));
  }

  // Create array from tensor
  std::vector<char> values;
  const auto elem_size = tensor.getElementSize();
  values.resize(elem_size * shape.numElements());
  char *values_ptr = values.data();
  ShapeRange out_range(shape);
  for (auto &out_idx : out_range)
  {
    std::memcpy(values_ptr, tensor.at(out_idx), elem_size);
    values_ptr += elem_size;
  }

  // Backslashes are not allowed in tensor names
  std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
  std::string filename = destination + "/" + tensor_name + ".hdf5";

  // Write to .hdf5 file
  H5::H5File h5File(filename, H5F_ACC_TRUNC);
  H5::DataSpace dataspace(rank, dims);
  H5::DataType h5_data_type;

  if (tensor.getDataType() == DataType::FLOAT32)
    h5_data_type = H5::PredType::NATIVE_FLOAT;
  else if (tensor.getDataType() == DataType::UINT8)
    h5_data_type = H5::PredType::NATIVE_UINT8;
  else
    throw std::runtime_error("NYI writing that DataType!");

  auto dataset = h5File.createDataSet(tensor_name, h5_data_type, dataspace);
  dataset.write(values.data(), h5_data_type);
}

#endif // NNC_HDF5_SUPPORTED

static TensorVariant readTensorFromFile(const std::string &filename, const TensorType &type)
{
  const std::size_t input_data_size =
      type.getShape().numElements() * getDataTypeSize(type.getElementType());

  std::ifstream stream(filename, std::ios::in | std::ios::binary);
  if (stream.fail())
    throw std::runtime_error("Couldn't open file \"" + filename + "\".");

  stream.seekg(0, std::ios::end);
  std::streampos end = stream.tellg();
  stream.seekg(0, std::ios::beg);
  std::streampos begin = stream.tellg();
  int64_t file_size = end - begin;

  if (static_cast<std::size_t>(file_size) != input_data_size)
    throw std::runtime_error("File \"" + filename +
                             "\" has incorrect size: " + std::to_string(file_size) +
                             "(expected: " + std::to_string(input_data_size) + ").");

  std::unique_ptr<char[]> data(new char[input_data_size]);
  stream.read(data.get(), input_data_size);
  if (stream.fail())
    throw std::runtime_error("Couldn't read file \"" + filename + "\".");

  return TensorVariant(type, data.get());
}

InterpreterBackend::InterpreterBackend(std::string input_dir, std::string output_dir)
    : _input_dir(std::move(input_dir)), _output_dir(std::move(output_dir))
{
}

void InterpreterBackend::run(mir::Graph *graph)
{
  assert(graph);

  mir_interpreter::MIRInterpreter interpreter;

  for (const auto *input_op : graph->getInputs())
  {
    const Operation::Output *input = input_op->getOutput(0);

    std::string tensor_name = input->getName();
    assert(!tensor_name.empty());
    std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
    std::string filename = _input_dir + "/" + tensor_name + ".dat";

    TensorVariant tensor = readTensorFromFile(filename, input->getType());
    interpreter.setTensor(input, std::move(tensor));
  }

  graph->accept(&interpreter);

  for (const auto *output_op : graph->getOutputs())
  {
    const auto &output_name = output_op->getInput(0)->getName();

#ifdef NNC_HDF5_SUPPORTED
    const auto &tensor = interpreter.getTensor(output_op->getInput(0));
    writeTensorToHDF5File(tensor, output_name, _output_dir);
#else
    std::cout << "Result <" << output_name << "> wasn't saved, due to lack of HDF5" << std::endl;
#endif // NNC_HDF5_SUPPORTED
  }
}

} // namespace nnc
