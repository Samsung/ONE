/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <H5Cpp.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

enum class ErrorCode
{
  CountMismatch,
  TypeMismatch,
  ShapeMismatch,
  ValueMismatch,
};

template <ErrorCode EC> class ErrorDetail;

// TODO Record the details
template <> class ErrorDetail<ErrorCode::CountMismatch>
{
public:
  ErrorDetail() = default;
};

// TODO Record the details
template <> class ErrorDetail<ErrorCode::TypeMismatch>
{
public:
  ErrorDetail() = default;
};

// TODO Record the details
template <> class ErrorDetail<ErrorCode::ShapeMismatch>
{
public:
  ErrorDetail() = default;
};

// TODO Record the details
template <> class ErrorDetail<ErrorCode::ValueMismatch>
{
public:
  ErrorDetail() = default;
};

struct Observer
{
  virtual ~Observer() = default;

  virtual void notify(const ErrorDetail<ErrorCode::CountMismatch> &) = 0;
  virtual void notify(const ErrorDetail<ErrorCode::TypeMismatch> &) = 0;
  virtual void notify(const ErrorDetail<ErrorCode::ShapeMismatch> &) = 0;
  virtual void notify(const ErrorDetail<ErrorCode::ValueMismatch> &) = 0;
};

class Mux final : public Observer
{
public:
  Mux() = default;

public:
  void attach(Observer *o) { _observers.insert(o); }

private:
  template <ErrorCode EC> void notify_all(const ErrorDetail<EC> &e)
  {
    for (auto o : _observers)
    {
      o->notify(e);
    }
  }

public:
  void notify(const ErrorDetail<ErrorCode::CountMismatch> &e) final { notify_all(e); }
  void notify(const ErrorDetail<ErrorCode::TypeMismatch> &e) final { notify_all(e); }
  void notify(const ErrorDetail<ErrorCode::ShapeMismatch> &e) final { notify_all(e); }
  void notify(const ErrorDetail<ErrorCode::ValueMismatch> &e) final { notify_all(e); }

public:
  std::set<Observer *> _observers;
};

class ExitcodeTracker final : public Observer
{
public:
  const int &exitcode(void) const { return _exitcode; }

public:
  void notify(const ErrorDetail<ErrorCode::CountMismatch> &) { _exitcode = 1; }
  void notify(const ErrorDetail<ErrorCode::TypeMismatch> &) { _exitcode = 1; }
  void notify(const ErrorDetail<ErrorCode::ShapeMismatch> &) { _exitcode = 1; }
  void notify(const ErrorDetail<ErrorCode::ValueMismatch> &) { _exitcode = 1; }

public:
  int _exitcode = 0;
};

} // namespace

//
// HDF5 helpers
//
namespace
{

enum class DataType
{
  UNKNOWN,
  FLOAT32,
  /* TO BE ADDED */
};

DataType to_internal_dtype(const H5::DataType &dtype)
{
  if (dtype == H5::PredType::IEEE_F32BE)
  {
    return DataType::FLOAT32;
  }
  return DataType::UNKNOWN;
}

using TensorShape = nncc::core::ADT::tensor::Shape;

TensorShape to_internal_shape(const H5::DataSpace &dataspace)
{
  int rank = dataspace.getSimpleExtentNdims();

  std::vector<hsize_t> dims;

  dims.resize(rank, 0);

  dataspace.getSimpleExtentDims(dims.data());

  TensorShape res;

  res.resize(rank);
  for (int axis = 0; axis < rank; ++axis)
  {
    res.dim(axis) = dims[axis];
  }

  return res;
}

uint32_t element_count(const H5::DataSpace &dataspace)
{
  return nncc::core::ADT::tensor::num_elements(to_internal_shape(dataspace));
}

std::vector<float> as_float_vector(const H5::DataSet &dataset)
{
  std::vector<float> buffer;

  buffer.resize(element_count(dataset.getSpace()));
  dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);

  return buffer;
}

using LexicalLayout = nncc::core::ADT::tensor::LexicalLayout;
using TensorIndexEnumerator = nncc::core::ADT::tensor::IndexEnumerator;

} // namespace

// TODO Report the details
int entry(int argc, char **argv)
{
  // The current implementation works only for command-line of the following form:
  //
  //   i5diff -d 0.001 /path/to/left.h5 /path/to/right.h5
  //
  // TODO Support more options
  assert(argc == 5);
  assert(std::string(argv[1]) == "-d");
  assert(std::string(argv[2]) == "0.001");

  H5::H5File lhs{argv[3], H5F_ACC_RDONLY};
  H5::H5File rhs{argv[4], H5F_ACC_RDONLY};

  ExitcodeTracker exitcode_tracker;

  Mux mux;
  mux.attach(&exitcode_tracker);

  // Compare values
  do
  {
    // NOTE The name of value group SHOULD BE aligned with nnkit HDF5 actions
    const std::string value_grpname{"value"};

    H5::Group lhs_value_grp = lhs.openGroup(value_grpname);
    H5::Group rhs_value_grp = rhs.openGroup(value_grpname);

    // Compare value count
    int64_t value_count = -1;
    {
      uint32_t lhs_value_count = static_cast<uint32_t>(lhs_value_grp.getNumObjs());
      uint32_t rhs_value_count = static_cast<uint32_t>(rhs_value_grp.getNumObjs());

      if (lhs_value_count != rhs_value_count)
      {
        ErrorDetail<ErrorCode::CountMismatch> error{};
        mux.notify(error);
        break;
      }

      value_count = std::max<int64_t>(lhs_value_count, rhs_value_count);
    }
    assert(value_count >= 0);

    // Compare each dataset
    for (int64_t n = 0; n < value_count; ++n)
    {
      // NOTE The name of dataset SHOULD BE aligned with nnkit HDF5 actions
      const std::string dataset_name = std::to_string(n);

      auto lhs_dataset = lhs_value_grp.openDataSet(dataset_name);
      auto rhs_dataset = rhs_value_grp.openDataSet(dataset_name);

      auto lhs_dtype = to_internal_dtype(lhs_dataset.getDataType());
      auto rhs_dtype = to_internal_dtype(rhs_dataset.getDataType());

      // TODO Support other data types
      assert(rhs_dtype == DataType::FLOAT32);
      assert(lhs_dtype == DataType::FLOAT32);

      if (lhs_dtype != rhs_dtype)
      {
        ErrorDetail<ErrorCode::TypeMismatch> error{};
        mux.notify(error);
        continue;
      }

      auto lhs_shape = to_internal_shape(lhs_dataset.getSpace());
      auto rhs_shape = to_internal_shape(rhs_dataset.getSpace());

      if (!(lhs_shape == rhs_shape))
      {
        ErrorDetail<ErrorCode::ShapeMismatch> error{};
        mux.notify(error);
        continue;
      }

      assert(lhs_shape == rhs_shape);
      assert(lhs_dtype == rhs_dtype);
      const auto &shape = lhs_shape;
      const auto &dtype = lhs_dtype;

      switch (dtype)
      {
        case DataType::FLOAT32:
        {
          auto lhs_vector = as_float_vector(lhs_dataset);
          auto rhs_vector = as_float_vector(rhs_dataset);

          assert(lhs_vector.size() == rhs_vector.size());

          LexicalLayout layout;

          for (TensorIndexEnumerator e{shape}; e.valid(); e.advance())
          {
            const auto &ind = e.current();
            auto lhs_value = lhs_vector.at(layout.offset(shape, ind));
            auto rhs_value = rhs_vector.at(layout.offset(shape, ind));

            // TODO Abstract equality criterion
            if (std::abs(lhs_value - rhs_value) >= 0.001f)
            {
              ErrorDetail<ErrorCode::ValueMismatch> error{};
              mux.notify(error);
              continue;
            }
          }

          break;
        }
        default:
          throw std::runtime_error{"Not supported, yet"};
      };
    }
  } while (false);

  // TODO Compare names (if requested)

  return exitcode_tracker.exitcode();
}
