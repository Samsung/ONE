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

#include "AclArtifact.h"
#include <iostream>
#include <memory>
#include <H5Cpp.h>

using namespace std;
using namespace arm_compute;

static unique_ptr<char[]> getTensorData(CLTensor &tensor)
{
  auto buf = unique_ptr<char[]>(new char[tensor.info()->total_size()]);
  tensor.map();
  Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());
  Iterator i(&tensor, window);
  char *ptr = &buf[0];

  execute_window_loop(
      window,
      [&i, &ptr](const Coordinates &) {
        memcpy(ptr, i.ptr(), sizeof(float));
        ptr += sizeof(float);
      },
      i);

  tensor.unmap();
  return buf;
}

static void readTensor(CLTensor &tensor, H5::DataSet &dataset)
{
  auto buf = unique_ptr<char[]>(new char[tensor.info()->total_size()]);
  dataset.read(&buf[0], H5::PredType::NATIVE_FLOAT);
  tensor.map();
  Window window;
  window.use_tensor_dimensions(tensor.info()->tensor_shape());
  Iterator i(&tensor, window);
  char *ptr = &buf[0];

  execute_window_loop(
      window,
      [&i, &ptr](const Coordinates &) {
        memcpy(i.ptr(), ptr, sizeof(float));
        ptr += sizeof(float);
      },
      i);

  tensor.unmap();
}

static bool readTensorFromHDF5File(CLTensor &tensor, const string &file_name)
{
  // Read from the .hdf5 file
  try
  {
    H5::H5File h5File(file_name, H5F_ACC_RDONLY);
    auto tensor_name = h5File.getObjnameByIdx(0);
    auto dataset = h5File.openDataSet(tensor_name);
    auto dataspace = dataset.getSpace();
    auto rank = dataspace.getSimpleExtentNdims();

    if (rank < 2)
      return false;

    hsize_t dims[rank];

    if (dataspace.getSimpleExtentDims(dims) != rank)
      return false;

    TensorShape shape;
    shape.set_num_dimensions(rank - 1);

    for (int i = 1; i < rank; ++i)
      shape[rank - i - 1] = dims[i];

    readTensor(tensor, dataset);
  }
  catch (H5::FileIException &)
  {
    return false;
  }

  return true;
}

static void writeTensorToHDF5File(CLTensor &tensor, const string &tensor_name,
                                  const string &file_name)
{
  const TensorShape &orig_shape = tensor.info()->tensor_shape();
  const TensorShape &transposed_shape = orig_shape;
  int rank = transposed_shape.num_dimensions();
  hsize_t dims[rank + 1];
  dims[0] = 1;

  for (int i = 0; i < rank; ++i)
    dims[rank - i] = transposed_shape[i];

  // Write to the .hdf5 file
  H5::H5File h5File(file_name, H5F_ACC_TRUNC);
  H5::DataSpace dataspace(rank + 1, dims);
  auto dataset = h5File.createDataSet(tensor_name, H5::PredType::IEEE_F32BE, dataspace);
  dataset.write(&getTensorData(tensor)[0], H5::PredType::NATIVE_FLOAT);
}

int main(int argc, char *argv[])
{
  CLScheduler::get().default_init();

  if (!CLScheduler::get().is_initialised())
  {
    cout << "Failed to initialise the ACL scheduler" << endl;
    return 1;
  }

  AclArtifact artifact;
  CLTensor &artifact_in = artifact.getInput();
  readTensorFromHDF5File(artifact_in, "in.hdf5");

  artifact.Inference();

  CLTensor &artifact_out = artifact.getOutput();
  writeTensorToHDF5File(artifact_out, "out", "out.hdf5");

  return 0;
}
