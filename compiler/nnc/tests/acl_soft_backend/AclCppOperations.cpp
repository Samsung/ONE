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

#include "gtest/gtest.h"
#include <sstream>
#include <thread>
#include <cmath>
#include <memory>
#include <H5Cpp.h>
#include <stdlib.h>
#include "BuildInfo.h"

using namespace std;

static string netAddr(getenv("ODROID_NET_ADDR") ? getenv("ODROID_NET_ADDR") : "");

static unique_ptr<char[]> readTensorDataFromHdf5File(const string &file_name, vector<int> &shape)
{
  try
  {
    H5::H5File h5File(file_name, H5F_ACC_RDONLY);
    auto tensor_name = h5File.getObjnameByIdx(0);
    auto dataset = h5File.openDataSet(tensor_name);
    auto dataspace = dataset.getSpace();
    auto rank = dataspace.getSimpleExtentNdims();

    if (rank < 2)
      return nullptr;

    hsize_t dims[rank];

    if (dataspace.getSimpleExtentDims(dims) != rank)
      return nullptr;

    int size = 1;

    for (int i = 0; i < rank; ++i)
    {
      size *= dims[i];
      shape.push_back(dims[i]);
    }

    auto result = unique_ptr<char[]>(new char[size * sizeof(float)]);
    dataset.read(&result[0], H5::PredType::NATIVE_FLOAT);
    return result;
  }
  catch (H5::FileIException &)
  {
    return nullptr;
  }
}

// TODO: this function was copied from CPPOperations.cpp, move it to a shared place.
bool areFloatsNear(float a, float b, int32_t ulp, float eps)
{
  assert(ulp < (1 << 23) && "this algorithm is not applicable for such large diffs");
  assert(eps >= 0 && "epsilon should be positive number");
  if (fabs(a - b) <= eps)
    return true;
  // since this point need to dind difference between numbers
  // in terms of ULP
  int32_t ai;
  int32_t bi;
  memcpy(&ai, &a, sizeof(float));
  memcpy(&bi, &b, sizeof(float));
  // compare mantissa of numbers
  if (ai > bi)
    return ai - bi <= ulp;
  return bi - ai <= ulp;
}

static void compareHdf5Files(const string &file_name1, const string &file_name2)
{
  vector<int> shape1;
  auto tensor1 = readTensorDataFromHdf5File(file_name1, shape1);
  float *tensorData1 = reinterpret_cast<float *>(&tensor1[0]);
  ASSERT_NE(tensorData1, nullptr);
  vector<int> shape2;
  auto tensor2 = readTensorDataFromHdf5File(file_name2, shape2);
  float *tensorData2 = reinterpret_cast<float *>(&tensor2[0]);
  ASSERT_NE(tensorData2, nullptr);
  ASSERT_EQ(shape1.size(), shape2.size());
  int size = 1;

  for (int i = 0; i < shape1.size(); ++i)
  {
    ASSERT_EQ(shape1[i], shape2[i]);
    size *= shape1[i];
  }

  for (int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(areFloatsNear(tensorData1[i], tensorData2[i], 32, 1e-6));
  }
}

static string genTmpDirName()
{
  string result("/tmp/nnc_test_");
  stringstream ss;
  ss << this_thread::get_id();
  result += ss.str();

  return result;
}

static bool runOnOdroid(const string &remote_cmd)
{
  string cmd = "ssh " + netAddr + " \"" + remote_cmd + "\"";
  return system(cmd.c_str()) == 0;
}

static bool copyToOdroid(const string &src, const string &dst)
{
  string cmd("scp -q " + src + " " + netAddr + ":" + dst);
  return system(cmd.c_str()) == 0;
}

static bool copyFromOdroid(const string &src, const string &dst)
{
  string cmd("scp -q " + netAddr + ":" + src + " " + dst);
  return system(cmd.c_str()) == 0;
}

static void runAclSystemTest(const string &name)
{
  // Ensure the Odroid device net address was set.
  ASSERT_TRUE(!netAddr.empty());

  // The name of the temporary directory which is generated on the remote device.
  string dir_name = genTmpDirName();

  // Insure there is no such the directory on the remote device.
  ASSERT_TRUE(runOnOdroid("rm -rf " + dir_name));

  // Create the temporary directory on the remote device.
  ASSERT_TRUE(runOnOdroid("mkdir " + dir_name));

  // Copy the executable artifact file to the remote device.
  ASSERT_TRUE(copyToOdroid(binDir + "/" + name + "/nnc_test", dir_name));

  // Copy the artifact parameter file to the remote device.
  ASSERT_TRUE(copyToOdroid(binDir + "/" + name + "/AclArtifact.par", dir_name));

  // Copy the model input HDF5 file to the remote device.
  ASSERT_TRUE(
    copyToOdroid(binDir + "/" + name + "/in_" + name + "_caffe.hdf5", dir_name + "/in.hdf5"));

  // Switch to the artifact directory on the remote device and run the artifact.
  ASSERT_TRUE(runOnOdroid("cd " + dir_name + "; ./nnc_test"));

  // Copy the resulting file from the remote device to the host.
  ASSERT_TRUE(copyFromOdroid(dir_name + "/out.hdf5", binDir + "/" + name));

  // Remove the temporary test case directory from the remote device.
  ASSERT_TRUE(runOnOdroid("rm -rf " + dir_name));

  // Compare the resulting HDF5 file with the reference one.
  compareHdf5Files(binDir + "/" + name + "/ref.hdf5", binDir + "/" + name + "/out.hdf5");
}

TEST(acl_cpp_operations_test, convolution) { runAclSystemTest("convolution"); }

TEST(acl_cpp_operations_test, depthwise_convolution) { runAclSystemTest("depthwise_convolution"); }

TEST(acl_cpp_operations_test, convolution_with_bias) { runAclSystemTest("convolution_with_bias"); }

TEST(acl_cpp_operations_test, scale) { runAclSystemTest("scale"); }

TEST(acl_cpp_operations_test, relu) { runAclSystemTest("relu"); }

TEST(acl_cpp_operations_test, pooling_max) { runAclSystemTest("pooling_max"); }

TEST(acl_cpp_operations_test, pooling_avg) { runAclSystemTest("pooling_avg"); }

TEST(acl_cpp_operations_test, concatenate) { runAclSystemTest("concatenate"); }

TEST(acl_cpp_operations_test, reshape) { runAclSystemTest("reshape"); }

TEST(acl_cpp_operations_test, fully_connected) { runAclSystemTest("fully_connected"); }
