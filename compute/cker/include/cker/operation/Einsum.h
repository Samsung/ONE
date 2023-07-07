/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_EINSUM_H__
#define __NNFW_CKER_EINSUM_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"

#include "cker/operation/Helper/Tensor.h"
#include "cker/operation/Helper/MatmulBCast.h"

#include "Transpose.h"
#include "BatchMatMul.h"

#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

namespace nnfw
{
namespace cker
{

namespace functor
{

template <typename Device, typename T, int N> struct StrideFunctor
{
  void operator()(const Device &d, typename TTypes<T, N>::ConstTensor input,
                  const std::vector<int32_t> &strides, typename TTypes<T, N>::Tensor output)
  {

    Eigen::DSizes<Eigen::DenseIndex, N> dsizes;
    for (size_t d = 0; d < strides.size(); d++)
    {
      dsizes[d] = static_cast<Eigen::DenseIndex>(strides[d]);
    }
    for (size_t d = strides.size(); d < N; d++)
    {
      dsizes[d] = 1;
    }

    output.device(d) = input.stride(dsizes);
  }
};

template <typename Device, typename T, int N> struct InflateFunctor
{
  void operator()(const Device &d, typename TTypes<T, N>::ConstTensor input,
                  const std::vector<int32_t> &strides, typename TTypes<T, N>::Tensor output)
  {

    Eigen::DSizes<Eigen::DenseIndex, N> dsizes;
    for (size_t d = 0; d < strides.size(); d++)
    {
      dsizes[d] = static_cast<Eigen::DenseIndex>(strides[d]);
    }
    for (size_t d = strides.size(); d < N; d++)
    {
      dsizes[d] = 1;
    }

    output.device(d) = input.inflate(dsizes);
  }
};

template <typename Device, typename Reducer> struct ReduceFunctor
{
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(const Device &d, OUT_T out, IN_T in, const ReductionAxes &reduction_axes,
                     const Reducer &reducer)
  {
    out.device(d) = in.reduce(reduction_axes, reducer);
  }
};

template <typename Device, typename T> struct SetZeroFunctor
{
  // Computes on device "d": out = out.setZero(),
  void operator()(const Device &d, typename TTypes<T>::Flat out)
  {
    out.device(d) = out.constant(T(0));
  }
};

} // namespace functor

using ShapeVec = std::vector<int32_t>;
using Labels = std::vector<int32_t>;
using OperandLabels = std::vector<Labels>;
using LabelCounts = std::vector<int32_t>;
using OperandLabelCounts = std::vector<LabelCounts>;
using LabelToDimSizes = std::vector<int32_t>;

// Each dimension is categorized into exactly one of five types based on
// whether its corresponding label is present in the input and/or the output
// subscripts.
enum DimensionType
{
  // Batch dimensions are those present in two inputs as well as the output.
  // They are part of the batch dimensions during Tensor contraction.
  // Such dimensions may be broadcasting dimensions (those mapping to
  // ellipsis)
  // or explicit batch dimensions corresponding to named axis labels.
  kBroadcasting = 0,
  kBatch = 1,
  // Free dimensions are present in exactly one of the inputs, and also the
  // output. These are non-contracted axes in the Tensor contraction.
  kFree = 2,
  // Contract dimensions are present in two inputs, but not the output. These
  // dimensions are contracted in Tensor contraction.
  kContract = 3,
  // Reduce dimensions are present in exactly one input; and not in the output
  // and are summed over prior to Tensor contraction.
  kReduce = 4,
};

namespace
{

constexpr int kEllipsisLabel = -1;

std::vector<std::string> strSplit(const std::string &text, const std::string delimiter)
{
  std::vector<std::string> result;

  size_t start = 0;
  size_t pos = 0;

  do
  {
    pos = text.find(delimiter, start);
    if (pos == std::string::npos)
    {
      result.push_back(text.substr(start, text.size() - start));
      break;
    }

    result.push_back(text.substr(start, pos - start));
    start = pos + delimiter.size();
  } while (pos != std::string::npos);

  return result;
}

inline DimensionType getDimensionType(bool is_removed, bool is_unique)
{
  if (!is_removed && !is_unique)
    return kBatch;
  else if (!is_removed && is_unique)
    return kFree;
  else if (is_removed && !is_unique)
    return kContract;
  else // is_removed && is_unique
    return kReduce;
}

inline Shape copyShape(const Shape &shape)
{
  return Shape::ExtendedShape(shape.DimensionsCount(), shape);
}
} // namespace

class Einsum
{
public:
  Einsum() : _prepared(false)
  {
    // DO NOTHING
  }

  void prepare(std::string &equation)
  {
    if (_prepared)
    {
      return;
    }

    // Parse equation
    parseEquation(equation);
    _prepared = true;
  }

  void operator()(std::string &equation, const std::vector<Shape> &input_shapes,
                  const std::vector<const float *> &input_data, const Shape &output_shape,
                  float *output_data)
  {
    if (!_prepared)
    {
      prepare(equation);
    }

    const int num_inputs = input_shapes.size();
    std::vector<InputTensor<float>> inputs(num_inputs);
    for (int i = 0; i < num_inputs; i++)
    {
      inputs[i].shape.ReplaceWith(input_shapes[i].DimensionsCount(), input_shapes[i].DimsData());
      inputs[i].buffer = input_data[i];
    }

    OperandLabels input_labels(_input_labels);
    Labels output_labels(_output_labels);
    std::vector<DimensionType> label_types(_label_types);
    OperandLabelCounts input_label_counts(_input_label_counts);
    LabelCounts output_label_counts(_output_label_counts);
    LabelToDimSizes label_to_dim_sizes;

    processDimensions(inputs, &input_labels, &output_labels, &label_types, &input_label_counts,
                      &output_label_counts, &label_to_dim_sizes);

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.

    OperandLabels free_labels(num_inputs);
    std::vector<Tensor> inputs_reduced(num_inputs);
    std::vector<bool> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i)
    {
      bool temp_swap_free_and_contract = false;
      reduceOperand<float>(inputs[i], label_types, input_label_counts[i], &input_labels[i],
                           &free_labels[i], &temp_swap_free_and_contract, &inputs_reduced[i]);
      swap_free_and_contract[i] = temp_swap_free_and_contract;
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    contractOperands(inputs_reduced, swap_free_and_contract, &contraction_output_reshaped);

    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    std::vector<int32_t> result_shape_dims(contraction_output_reshaped.shape.DimensionsCount() - 2);

    for (size_t i = 0; i < result_shape_dims.size(); i++)
    {
      result_shape_dims[i] = contraction_output_reshaped.shape.Dims(i);
    }

    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label)
    {
      if (label_types[label] == kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label)
    {
      if (label_types[label] == kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i)
    {
      for (auto &&label : free_labels[i])
      {
        result_labels.push_back(label);
        result_shape_dims.push_back(label_to_dim_sizes[label]);
      }
    }

    Shape result_shape(result_shape_dims.size(), result_shape_dims.data());

    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    copyFrom(contraction_output_reshaped, result_shape, &contraction_output);

    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    // TODO(anudhyan): It's possible that Eigen's contract and inflate can be
    // chained here to avoid materializing an intermediate.
    Tensor output_inflated;
    strideOrInflate<float>(contraction_output, result_labels, output_label_counts,
                           true /* should_inflate */, &output_inflated);

    if (output_inflated.shape.DimensionsCount() > contraction_output.shape.DimensionsCount())
    {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (auto &&label : result_labels)
      {
        inflated_labels.insert(inflated_labels.end(), output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }

    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int32_t> output_permutation(output_labels.size());
    std::vector<int32_t> label_to_position(num_labels, -1);
    for (size_t i = 0; i < result_labels.size(); ++i)
    {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1)
      {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (size_t i = 0; i < output_labels.size(); ++i)
    {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }

    InputTensor<float> temp_inflated;
    temp_inflated.shape.ReplaceWith(output_inflated.shape.DimensionsCount(),
                                    output_inflated.shape.DimsData());
    temp_inflated.buffer = (reinterpret_cast<const float *>(output_inflated.buffer));
    ;

    Tensor output;
    transposeOperand<float>(temp_inflated, output_permutation, &output);

    memcpy(output_data, output.buffer, output_shape.FlatSize() * sizeof(float));

    temp_operand.clear();
  }

private:
  void parseEquation(std::string &equation)
  {
    std::vector<std::string> input_str;
    std::string output_str;

    parseEinsumEquation(equation, input_str, output_str);

    // Temporary map from single character labels to (consecutive) integer
    // labels.
    std::map<char, int> label_mapping;
    int num_inputs = input_str.size();
    _input_labels.resize(num_inputs);

    // Map from single characters to integer labels.
    for (int i = 0; i < num_inputs; ++i)
    {
      mapToLabels(input_str[i], _input_labels.at(i), label_mapping);
    }
    mapToLabels(output_str, _output_labels, label_mapping);

    // Compute counts for input and output labels.
    int num_labels = label_mapping.size();
    _input_label_counts.resize(num_inputs);
    _input_has_ellipsis.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i)
    {
      _input_label_counts.at(i).resize(num_labels);
      for (const int label : _input_labels.at(i))
      {
        if (label != kEllipsisLabel)
          _input_label_counts.at(i)[label] += 1;
        else
          _input_has_ellipsis.at(i) = true;
      }
    }
    _output_label_counts.resize(num_labels);
    for (const int label : _output_labels)
    {
      if (label != kEllipsisLabel)
        _output_label_counts.at(label) += 1;
      else
        _output_has_ellipsis = true;
    }

    // Map each label to a unique DimensionType.
    _label_types.resize(num_labels);
    for (int label = 0; label < num_labels; ++label)
    {
      bool removed = (_output_label_counts[label] == 0);
      bool unique =
        num_inputs == 1 || _input_label_counts[0][label] == 0 || _input_label_counts[1][label] == 0;
      _label_types[label] = getDimensionType(removed, unique);
    }
  }

  void parseEinsumEquation(const std::string &equation, std::vector<std::string> &input_subscripts,
                           std::string &output_subscript)
  {
    std::vector<std::string> inputs_and_output_subscripts = strSplit(equation, "->");
    if (inputs_and_output_subscripts.size() != 2)
    {
      throw std::runtime_error{"Einsum: Expecting exactly one '->' in einsum equation: " +
                               equation};
    }

    output_subscript = inputs_and_output_subscripts[1];
    input_subscripts = strSplit(inputs_and_output_subscripts[0], ",");
    if (input_subscripts.size() != 1 && input_subscripts.size() != 2)
    {
      throw std::runtime_error{"Einsum: Expecting 1 or 2 input subscripts in equation '" +
                               equation + "' but got: " + std::to_string(input_subscripts.size())};
    }
  }

  // Maps the character labels to consecutive integers.
  void mapToLabels(const std::string &subscript, Labels &labels, std::map<char, int> &label_mapping)
  {
    for (size_t i = 0; i < subscript.size(); ++i)
    {
      const char label_char = subscript[i];
      if (label_char == '.')
      {
        labels.push_back(kEllipsisLabel);
        i += 2; // Skip next 2 characters as well.
        continue;
      }
      if (label_mapping.find(label_char) == label_mapping.end())
      {
        const int next_label = label_mapping.size();
        label_mapping[label_char] = next_label;
      }
      const int mapped_label = label_mapping[label_char];
      labels.push_back(mapped_label);
    }
  }

  template <typename T>
  void processDimensions(const std::vector<InputTensor<T>> &inputs, OperandLabels *input_labels,
                         Labels *output_labels, std::vector<DimensionType> *label_types,
                         OperandLabelCounts *input_label_counts, LabelCounts *output_label_counts,
                         LabelToDimSizes *label_to_dim_sizes)
  {
    if (inputs.size() != input_labels->size())
    {
      throw std::runtime_error{"Expected " + std::to_string(input_labels->size()) +
                               " inputs but got: " + std::to_string(inputs.size())};
    }
    const int num_inputs = inputs.size();

    // We infer the number of broadcasting dimensions by taking the maximum rank
    // among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;
    const int num_named_labels = label_types->size();
    label_to_dim_sizes->resize(num_named_labels);
    for (int i = 0; i < num_inputs; ++i)
    {
      Labels *labels = &(*input_labels)[i];

      if (!_input_has_ellipsis[i])
      {
        if (inputs[i].shape.DimensionsCount() != ((int32_t)labels->size()))
        {
          throw std::runtime_error{"Expected input " + std::to_string(i) + " to have rank " +
                                   std::to_string(labels->size()) + " but got: " +
                                   std::to_string(inputs[i].shape.DimensionsCount())};
        }
        for (size_t label_idx = 0; label_idx < labels->size(); ++label_idx)
        {
          const int label = (*labels)[label_idx];
          recordLabelToDimension(label, label_idx, inputs[i].shape, label_to_dim_sizes);
        }
        continue;
      }

      // Input has an ellipsis.
      if (inputs[i].shape.DimensionsCount() + 1 < (int32_t)labels->size())
      {
        throw std::runtime_error{"Expected input " + std::to_string(i) + " to have rank at least " +
                                 std::to_string(labels->size() - 1) +
                                 " but got: " + std::to_string(inputs[i].shape.DimensionsCount())};
      }
      int ellipsis_axis = -1;
      const int num_bcast_dims = inputs[i].shape.DimensionsCount() - labels->size() + 1;
      for (size_t label_idx = 0; label_idx < labels->size(); ++label_idx)
      {
        const int label = (*labels)[label_idx];
        if (label == kEllipsisLabel)
        {
          ellipsis_axis = label_idx;
          continue;
        }
        // Current label is not an ellipsis.
        const int axis = label_idx + (ellipsis_axis == -1 ? 0 : num_bcast_dims - 1);
        recordLabelToDimension(label, axis, inputs[i].shape, label_to_dim_sizes);
      }
      // Found an ellipsis. Replace 'kEllipsisLabel' with broadcasting
      // dimensions.
      if (ellipsis_axis != -1)
      {
        insertBroadcastLabels(num_bcast_dims, num_named_labels, ellipsis_axis, labels,
                              &input_label_counts->at(i));
        max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
      }
    }

    std::vector<bool>::iterator it_input =
      std::find(_input_has_ellipsis.begin(), _input_has_ellipsis.end(), true);
    if (it_input == _input_has_ellipsis.end() && !_output_has_ellipsis)
    {
      return;
    }
    // Insert broadcasting dimensions in the output labels.
    auto it = std::find(output_labels->begin(), output_labels->end(), kEllipsisLabel);
    if (it != output_labels->end())
    {
      const int ellipsis_axis = it - output_labels->begin();
      insertBroadcastLabels(max_bcast_dims, num_named_labels, ellipsis_axis, output_labels,
                            output_label_counts);
    }
    else if (max_bcast_dims > 0)
    {
      std::runtime_error{"Output contains " + std::to_string(max_bcast_dims) +
                         " broadcasting dimension(s) but no ellipsis " +
                         "(...) was found in the output subscripts."};
    }
    // Populate DimensionType for the new broadcasting labels.
    label_types->resize(num_named_labels + max_bcast_dims, kBroadcasting);
  }

  void recordLabelToDimension(const int32_t label, const int axis, const Shape &input_shape,
                              LabelToDimSizes *label_to_dim_sizes)
  {
    const int32_t input_dim = input_shape.Dims(axis);
    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label_to_dim_sizes->at(label) != 0 && label_to_dim_sizes->at(label) != input_dim)
    {
      std::runtime_error{"Expected dimension " + std::to_string(label_to_dim_sizes->at(label)) +
                         " at axis " + std::to_string(axis) +
                         " of the input shaped but got dimension " + std::to_string(input_dim)};
    }
    (*label_to_dim_sizes)[label] = input_dim;
  }

  void insertBroadcastLabels(int num_bcast_dims, int num_named_labels, int ellipsis_axis,
                             Labels *labels, LabelCounts *label_counts)
  {
    labels->erase(labels->begin() + ellipsis_axis);
    labels->insert(labels->begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels->begin() + ellipsis_axis, labels->begin() + ellipsis_axis + num_bcast_dims,
              num_named_labels);
    // Increment label counts. Since these are new labels, the count is set
    // to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
  }

  template <typename T>
  void reduceOperand(const InputTensor<T> &input, const std::vector<DimensionType> &label_types,
                     const LabelCounts &label_counts, Labels *labels, Labels *free_labels,
                     bool *swap_free_and_contract, Tensor *output)
  {
    // Find the permutation to transpose the input dimensions in the order of
    // DimensionType; i.e. batch, free, contract and reduce dimensions. This
    // makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int32_t> permutation(input.shape.DimensionsCount());
    std::iota(permutation.begin(), permutation.end(), 0);
    Tensor input_transposed;

    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if (shouldSwapFreeAndContract(*labels, label_types))
    {
      *swap_free_and_contract = true;
    }
    else
    {
      std::sort(permutation.begin(), permutation.end(), [&](int i, int j) {
        int label_i = (*labels)[i];
        int label_j = (*labels)[j];
        return std::tie(label_types[label_i], label_i) < std::tie(label_types[label_j], label_j);
      });
    }
    // Transpose the input so that DimensionTypes are in order.
    transposeOperand<T>(input, permutation, &input_transposed);

    permuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    Tensor input_deduped;
    labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    strideOrInflate<T>(input_transposed, *labels, label_counts, false /* should_inflate */,
                       &input_deduped);

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each DimensionType.
    std::vector<int32_t> reshape(5, 1);

    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    Shape output_shape;
    std::vector<int32_t> output_shape_dims;
    for (size_t label_idx = 0; label_idx < labels->size(); ++label_idx)
    {
      const int label = labels->at(label_idx);
      int32_t dim = input_deduped.shape.Dims(label_idx);
      if (label_types[label] == kBroadcasting || label_types[label] == kBatch)
      {
        output_shape_dims.push_back(dim);
      }
      else if (label_types[label] == kFree)
      {
        free_labels->push_back(label);
      }
      reshape[label_types[label]] *= dim;
    }

    if (*swap_free_and_contract)
      std::swap(reshape[kFree], reshape[kContract]);

    output_shape_dims.push_back(reshape[kFree]);
    output_shape_dims.push_back(reshape[kContract]);

    output_shape.ReplaceWith(output_shape_dims.size(), output_shape_dims.data());

    if (reshape[kReduce] == 1)
    { // No need to actually reduce.
      return copyFrom(input_deduped, output_shape, output);
    }

    allocateTemp(output_shape, output);

    using Reducer = Eigen::internal::SumReducer<T>;
    using Index = typename TTypes<T>::Tensor::Index;

    const Eigen::ThreadPoolDevice &device = *eigen_support::GetThreadPoolDevice();

    // Reduce along the last axis (i.e axis 1) of the rank-2 Tensor.
    const int32_t output_size =
      reshape[kBroadcasting] * reshape[kBatch] * reshape[kFree] * reshape[kContract];
    functor::ReduceFunctor<Eigen::ThreadPoolDevice, Reducer>::Reduce(
      device, output->shaped<T, 1>({output_size}),
      input_deduped.shaped<T, 2>({output_size, reshape[kReduce]}), Eigen::array<Index, 1>({1}),
      Reducer());
  }

  bool shouldSwapFreeAndContract(const Labels &labels,
                                 const std::vector<DimensionType> &label_types)
  {
    // Check that ordering is according to dimension type, with the role of
    // free and contract dimensions swapped.
    std::vector<int> remap = {0, 1, 3, 2, 4};
    for (size_t i = 0; i + 1 < labels.size(); ++i)
    {
      const int dimtype_a = remap[label_types[labels[i]]];
      const int dimtype_b = remap[label_types[labels[i + 1]]];
      if (dimtype_a > dimtype_b || (dimtype_a == dimtype_b && labels[i] > labels[i + 1]))
      {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  void transposeOperand(const InputTensor<T> &input, const std::vector<int32_t> &permutation,
                        Tensor *output)
  {
    if (!shouldTranspose(input.shape, permutation))
    {
      copyFrom(input, input.shape, output);
      return;
    }
    Shape transposed_shape(input.shape.DimensionsCount());
    for (int i = 0; i < input.shape.DimensionsCount(); ++i)
    {
      transposed_shape.SetDim(i, input.shape.Dims(permutation[i]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.shape.FlatSize() == 0)
    {
      copyFrom(input, transposed_shape, output);
      return;
    }

    temp_operand.emplace_back(std::make_unique<T[]>(transposed_shape.FlatSize()));
    T *new_buffer = temp_operand.back().get();

    TransposeParams transpose_params;
    transpose_params.perm_count = permutation.size();
    for (size_t i = 0; i < permutation.size(); i++)
    {
      transpose_params.perm[i] = permutation[i];
    }

    Transpose<T>(transpose_params, input.shape, input.buffer, transposed_shape, new_buffer);

    output->shape.ReplaceWith(transposed_shape.DimensionsCount(), transposed_shape.DimsData());
    output->buffer = new_buffer;
  }

  bool shouldTranspose(const Shape &input_shape, const std::vector<int32_t> &permutation)
  {
    if (input_shape.DimensionsCount() < 2)
      return false;
    for (size_t i = 0; i < permutation.size(); ++i)
    {
      if (permutation[i] != (int32_t)i)
        return true;
    }
    return false;
  }

  template <typename T>
  void copyFrom(const InputTensor<T> &input, const Shape &shape, Tensor *output)
  {
    Tensor temp_tensor;
    temp_tensor.shape.ReplaceWith(input.shape.DimensionsCount(), input.shape.DimsData());
    temp_operand.emplace_back(std::make_unique<float[]>(input.shape.FlatSize()));
    temp_tensor.buffer = temp_operand.back().get();
    memcpy(temp_tensor.buffer, input.buffer, input.shape.FlatSize() * sizeof(float));

    copyFrom(temp_tensor, shape, output);
  }

  void copyFrom(const Tensor &input, const Shape &shape, Tensor *output)
  {
    if (output->copyFrom(input, shape))
      return;

    throw std::runtime_error{"Einsum: Encountered error while reshaping a Tensor"};
  }

  // Permutes the labels according to the given permutation.
  void permuteLabels(const std::vector<int32_t> &permutation, Labels *labels)
  {
    Labels permuted_labels(labels->size());
    for (size_t i = 0; i < labels->size(); ++i)
    {
      permuted_labels[i] = (*labels)[permutation[i]];
    }
    labels->swap(permuted_labels);
  }

  // If there are repeated labels in either the input or output, then this
  // strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
  template <typename T>
  void strideOrInflate(const Tensor &input, const Labels &labels, const LabelCounts &label_counts,
                       const bool should_inflate, Tensor *output)
  {
    // Return early if there are no repeated indices.
    if (std::all_of(label_counts.begin(), label_counts.end(), [](int c) { return c <= 1; }))
    {
      return copyFrom(input, input.shape, output);
    }
    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    std::vector<int32_t> reshape;
    std::vector<int32_t> strides;
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    Shape strided_shape;
    Shape inflated_shape;
    std::vector<int32_t> strided_shape_dims;
    std::vector<int32_t> inflated_shape_dims;
    for (auto &&label : labels)
    {
      const int32_t count = label_counts[label];
      const int current_axis =
        should_inflate ? strided_shape_dims.size() : inflated_shape_dims.size();
      const int32_t dim = input.shape.Dims(current_axis);
      strided_shape_dims.push_back(dim);
      inflated_shape_dims.insert(inflated_shape_dims.end(), count, dim);
      const int32_t reshape_dim = std::pow(dim, count);
      reshape.push_back(reshape_dim);
      // While taking the d-diagonal in a rank k Tensor, we take d
      // equally-spaced elements including the first and last element. Then, (k
      // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
      const int32_t stride = (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
      strides.push_back(stride);
    }

    strided_shape.ReplaceWith(strided_shape_dims.size(), strided_shape_dims.data());
    inflated_shape.ReplaceWith(inflated_shape_dims.size(), inflated_shape_dims.data());

    Shape output_shape = Shape(should_inflate ? inflated_shape : strided_shape);

    output->shape.ReplaceWith(output_shape.DimensionsCount(), output_shape.DimsData());
    temp_operand.emplace_back(std::make_unique<float[]>(output_shape.FlatSize()));
    output->buffer = temp_operand.back().get();

    const Eigen::ThreadPoolDevice &device = *eigen_support::GetThreadPoolDevice();

    switch (reshape.size())
    {
#define NDIMS_CASE(N)                                                                      \
  case N:                                                                                  \
  {                                                                                        \
    if (should_inflate)                                                                    \
    {                                                                                      \
      auto output_map = output->shaped<T, N>(reshape);                                     \
      auto input_map = input.shaped<T, N>(strided_shape_dims);                             \
      functor::InflateFunctor<Eigen::ThreadPoolDevice, T, N>()(device, input_map, strides, \
                                                               output_map);                \
    }                                                                                      \
    else                                                                                   \
    {                                                                                      \
      auto input_map = input.shaped<T, N>(reshape);                                        \
      auto output_map = output->shaped<T, N>(strided_shape_dims);                          \
      functor::StrideFunctor<Eigen::ThreadPoolDevice, T, N>()(device, input_map, strides,  \
                                                              output_map);                 \
    }                                                                                      \
  }                                                                                        \
  break;
      NDIMS_CASE(1);
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      NDIMS_CASE(6);
      default:
        throw std::runtime_error{"Unsupported rank: " + std::to_string(reshape.size()) +
                                 " while handling repeated indices. Up to rank 6 is supported."};
#undef NDIMS_CASE
    }
  }

  void allocateTemp(const Shape &shape, Tensor *output)
  {
    output->shape.ReplaceWith(shape.DimensionsCount(), shape.DimsData());
    temp_operand.emplace_back(std::make_unique<float[]>(shape.FlatSize()));
    output->buffer = temp_operand.back().get();
  }

  // Contracts the inputs along the last axis. (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.
  // TODO(anudhyan): Factor this function into a BatchMatMul functor and support
  // transpose_x and transpose_y attributes (in addition to adj_x and adj_y).
  // Also, the BatchMatMul might devolve into a component-wise multiplication
  // when the matrix shape is [1,1]; in this case BatchMatMul functor would be
  // very inefficient. The functor should detect if this is the case and perform
  // componentwise multiplication functor instead.
  void contractOperands(std::vector<Tensor> &inputs, std::vector<bool> &swap_free_and_contract,
                        Tensor *output)
  {
    if (inputs.size() == 1)
      return copyFrom(inputs[0], inputs[0].shape, output);

    MatMulBCast bcast(inputs[0].shape, inputs[1].shape);
    if (!bcast.IsValid())
    {
      throw std::runtime_error{"Einsum: Invalid broadcasting dimensions"};
    }

    Tensor lhs;
    reshapeToRank3(inputs[0], bcast.x_batch_size(), &lhs);
    Tensor rhs;
    reshapeToRank3(inputs[1], bcast.y_batch_size(), &rhs);
    Shape old_output_shape = bcast.output_batch_shape();
    Shape output_shape(old_output_shape.DimensionsCount() + inputs.size());
    for (int i = 0; i < old_output_shape.DimensionsCount(); i++)
    {
      output_shape.SetDim(i, old_output_shape.Dims(i));
    }

    for (size_t i = 0; i < inputs.size(); ++i)
    {
      const int32_t free_axis =
        inputs[i].shape.DimensionsCount() - (swap_free_and_contract[i] ? 1 : 2);
      output_shape.SetDim(i + old_output_shape.DimensionsCount(), inputs[i].shape.Dims(free_axis));
    }
    bool adj_x = swap_free_and_contract[0];
    bool adj_y = !swap_free_and_contract[1];

    allocateTemp(output_shape, output);

    const Eigen::ThreadPoolDevice &device = *eigen_support::GetThreadPoolDevice();

    if (lhs.shape.FlatSize() == 0 || rhs.shape.FlatSize() == 0)
    {
      functor::SetZeroFunctor<Eigen::ThreadPoolDevice, float> set_zero;
      set_zero(device,
               typename TTypes<float, 1>::Tensor(output->base<float>(), output->shape.FlatSize()));
      return;
    }

    Tensor output_reshaped;
    reshapeToRank3(*output, bcast.output_batch_size(), &output_reshaped);

    // LaunchBatchMatMul::Launch(lhs, rhs, adj_x, adj_y, bcast, &output_reshaped);
    BatchMatMul batchMatMul;
    batchMatMul.prepare(lhs.shape, rhs.shape, adj_x, adj_y);
    batchMatMul(lhs.shape, lhs.base<float>(), rhs.shape, rhs.base<float>(), adj_x, adj_y,
                output_reshaped.shape, output_reshaped.base<float>());
  }

  void reshapeToRank3(const Tensor &input, int batch_size, Tensor *output)
  {
    const int rank = input.shape.DimensionsCount();
    Shape output_shape({batch_size, input.shape.Dims(rank - 2), input.shape.Dims(rank - 1)});
    copyFrom(input, output_shape, output);
  }

private:
  bool _prepared;

  OperandLabels _input_labels;
  Labels _output_labels;
  std::vector<DimensionType> _label_types;
  OperandLabelCounts _input_label_counts;
  LabelCounts _output_label_counts;
  std::vector<bool> _input_has_ellipsis;
  bool _output_has_ellipsis = false;

  std::vector<std::unique_ptr<float[]>> temp_operand;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EINSUM_H__
