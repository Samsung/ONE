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

#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <H5Cpp.h>

using namespace std;

class Tensor;

static void iterate(Tensor &tensor, function<void(vector<int> &)> on_loop);

class Tensor
{
public:
  explicit Tensor(const vector<hsize_t> &shape) : _shape(shape), _data(0), _num_elems(1)
  {
    _strides.resize(shape.size());

    for (int i = _shape.size() - 1; i >= 0; --i)
    {
      _strides[i] = _num_elems;
      _num_elems *= _shape[i];
    }

    _data = new float[_num_elems];
  }

  ~Tensor() { delete[] _data; }
  int rank() const { return _shape.size(); }
  int dim(int d) const { return _shape[d]; }
  float *data() { return _data; }
  float numElems() const { return _num_elems; }

  float &at(const vector<int> &coords)
  {
    int offset = 0;

    for (auto i = 0; i < coords.size(); ++i)
      offset += coords[i] * _strides[i];

    return _data[offset];
  }

  Tensor transpose(const vector<hsize_t> &reshape)
  {
    vector<hsize_t> tr_shape(_shape.size());

    for (auto i = 0; i < _shape.size(); ++i)
      tr_shape[i] = _shape[reshape[i]];

    Tensor result(tr_shape);
    auto on_loop = [this, &reshape, &result](vector<int> &coords) {
      vector<int> tr_coords(_shape.size());

      for (int i = 0; i < rank(); ++i)
        tr_coords[i] = coords[reshape[i]];

      result.at(tr_coords) = at(coords);
    };
    iterate(*this, on_loop);
    return result;
  }

private:
  vector<hsize_t> _shape;
  vector<hsize_t> _strides;
  float *_data;
  hsize_t _num_elems;
};

static void fillTensor(Tensor &tensor)
{
  int v = 10;

  for (int i = 0; i < tensor.numElems(); ++i)
  {
    tensor.data()[i] = v;
    v += 10;
  }
}

static void iterate(Tensor &tensor, function<void(vector<int> &)> on_loop)
{
  int num_dims = tensor.rank();
  vector<int> coords(num_dims, 0);
  vector<int> dims(num_dims);

  for (int i = 0; i < num_dims; ++i)
    dims[i] = tensor.dim(i);

  for (;;)
  {
    on_loop(coords);

    int i;
    for (i = num_dims - 1; i >= 0; --i)
    {
      if (coords[i] < dims[i] - 1)
      {
        ++coords[i];
        break;
      }
    }

    if (i < 0)
      break;

    fill(coords.begin() + i + 1, coords.end(), 0);
  }
}

static void dumpTensor(Tensor &tensor)
{
  auto on_loop = [&tensor](vector<int> &coords) {
    for (int i = 0; i < tensor.rank(); ++i)
    {
      if (i > 0)
        cout << ", ";

      cout << coords[i];
    }

    cout << " = " << tensor.at(coords) << endl;
  };

  iterate(tensor, on_loop);
}

static void writeTensorToDatFile(const string &file_name, Tensor &tensor)
{
  ofstream of(file_name + ".dat", ios_base::binary);

  if (of.fail())
    cout << "Could not output tensor to the: " << file_name + ".dat";

  of.write(reinterpret_cast<char *>(tensor.data()), tensor.numElems() * sizeof(float));
}

static void writeTensorToHDF5File(const vector<hsize_t> &dimensions, const string &tensor_name,
                                  const string &file_name, Tensor &tensor)
{
  H5::H5File h5File(file_name + ".hdf5", H5F_ACC_TRUNC);
  H5::DataSpace dataspace(dimensions.size(), &dimensions[0]);
  auto dataset = h5File.createDataSet(tensor_name, H5::PredType::IEEE_F32BE, dataspace);
  dataset.write(tensor.data(), H5::PredType::NATIVE_FLOAT);
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    cout << "Usage: " << argv[0] << " <tensor name> <output file name> dim0, dim1, dim2, ..."
         << endl;
    cout << "Where: dim0, dim1, dim2, ... are the generated tensor shape dimensions" << endl;
    return 1;
  }

  vector<hsize_t> dimensions;

  for (int i = 3; i < argc; ++i)
  {
    try
    {
      int d = stoi(argv[i]);

      if (d <= 0)
      {
        cout << "The dimensions must be positive values. This is not a correct dimension value: "
             << d << endl;
        return 1;
      }

      dimensions.push_back(d);
    }
    catch (const invalid_argument &)
    {
      cout << "The parameter does not look as an integer value: " << argv[i] << endl;
      return 1;
    }
    catch (const out_of_range &)
    {
      cout << "The value is out of the C++ \"int\" type range: " << argv[i] << endl;
      return 1;
    }
  }

  Tensor caffe_tensor(dimensions);
  fillTensor(caffe_tensor);
  writeTensorToHDF5File(dimensions, argv[1], "in_" + string(argv[2]) + "_caffe", caffe_tensor);

  vector<hsize_t> tf_reshape{0, 2, 3, 1};
  Tensor tf_tensor = caffe_tensor.transpose(tf_reshape);
  writeTensorToDatFile(string(argv[2]) + "_tf", tf_tensor);

  return 0;
}
