/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef __NNFW_CKER_EIGEN_REDUX_FUNCTOR_H__
#define __NNFW_CKER_EIGEN_REDUX_FUNCTOR_H__

#include <cker/operation/Helper/Tensor.h>

// From tensorflow/core/kernels/redux_functor.h
namespace nnfw
{
namespace cker
{
namespace functor
{

// Compute reduction over outer dimensions.
// Example:
//   input: [D1, D2, ... , DN]
//   ->
//   output: [Di, ... , DN] where i belongs to set [1,N]
template <typename Device, typename InputT, typename AccumT, typename OutputT,
          typename BinaryFunctor>
struct ReduceOuterDimensions
{
  ReduceOuterDimensions() {}

  template <int num_dims>
  void operator()(const Device &device, const Eigen::DSizes<Eigen::Index, num_dims> &input_dims,
                  const Tensor &input, Tensor *output) const
  {
    // Compute inner and outer dim after reshaping into 2d tensor.
    const int num_output_dims = output->shape.DimensionsCount();
    auto output_dims = output->template flat<OutputT>().dimensions();

    Eigen::Index inner_dim = 1, outer_dim = 1;
    for (int i = 0; i < num_dims - num_output_dims; ++i)
      outer_dim *= input_dims[i];
    for (int i = num_dims - num_output_dims; i < num_dims; ++i)
      inner_dim *= input_dims[i];

    if (1 == outer_dim)
    {
      // Nothing to do but passing input to output.
      output->template flat<OutputT>() =
        input.template flat<InputT>().template cast<OutputT>().reshape(output_dims);
      return;
    }

    // Get device thread num.
    const Eigen::Index num_threads = device.numThreads();

    // If the inner dim parallelism is large enough
    // TODO(ezhulenev): There seems to be no benefits in going this route. Check
    // if this can be improved, or use better heuristic?
    if (inner_dim > num_threads * 32)
    {
      // Do not create more blocks than there are threads in a pool.
      const Eigen::Index num_blocks = num_threads;

      // Block size along the outer dimension.
      const Eigen::Index inner_block_size = Eigen::divup(inner_dim, num_blocks);
      const InputT *input_data = input.template flat<InputT>().data();

      // Allocate temporary buffer for partial reductions.
      Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index> buffer({inner_dim});
      buffer.setZero();
      AccumT *buffer_data = buffer.data();

      using Buffer =
        Eigen::TensorMap<Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index>, Eigen::Unaligned>;

      using Input = Eigen::TensorMap<Eigen::Tensor<const InputT, 1, Eigen::RowMajor, Eigen::Index>,
                                     Eigen::Unaligned>;

      const auto compute = [inner_dim, outer_dim, inner_block_size, input_data,
                            buffer_data](Eigen::Index start, Eigen::Index limit) -> void {
        Eigen::Index inner_dim_start = start * inner_block_size;
        Eigen::Index inner_dim_limit = limit * inner_block_size;
        inner_dim_limit = std::min(inner_dim, inner_dim_limit);
        Eigen::Index my_job_len = inner_dim_limit - inner_dim_start;

        const InputT *my_job_start = input_data + inner_dim_start;
        Buffer buf(buffer_data + inner_dim_start, my_job_len);

        for (Eigen::Index i = 0; i < outer_dim; ++i)
        {
          auto in = Input(my_job_start + i * inner_dim, my_job_len);
          auto cast = in.template cast<AccumT>();
          buf =
            Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf), const decltype(cast)>(
              buf, cast);
        }
      };

      // Compute cost of reducing a single block.
      const Eigen::Index compute_size = outer_dim * inner_block_size;
      const Eigen::Index compute_input_bytes = compute_size * sizeof(InputT);
      const Eigen::TensorOpCost cost(compute_input_bytes,
                                     0, // We'll be mostly writing to L1, assume store cost is 0
                                     compute_size *
                                       Eigen::internal::functor_traits<BinaryFunctor>::Cost);

      device.parallelFor(num_blocks, cost, compute);

      // Write final result to the output.
      output->template flat<OutputT>() = buffer.template cast<OutputT>().reshape(output_dims);
    }
    else
    {
      // Compute block size along the outer dimension for efficiency.
      const Eigen::Index parallel_cell_size = inner_dim;
      const Eigen::Index total_workload = outer_dim * inner_dim;
      const Eigen::Index max_parallelism = total_workload / parallel_cell_size;

      const Eigen::Index min_block_workload = 2000;
      const Eigen::Index min_block_size = Eigen::divup(min_block_workload, parallel_cell_size);
      const Eigen::Index max_num_blocks =
        std::min(max_parallelism, Eigen::divup(total_workload, min_block_size));

      // Do not create more blocks than there are threads in a pool.
      const Eigen::Index num_blocks = std::min(max_num_blocks, num_threads);

      // Block size along the outer dimension.
      const Eigen::Index outer_block_size = Eigen::divup(outer_dim, num_blocks);

      const InputT *input_data = input.template flat<InputT>().data();

      // Allocate temporary buffer for partial reductions.
      std::vector<AccumT> buffer(num_blocks * inner_dim);
      AccumT *buffer_data = buffer.data();

      using Buffer =
        Eigen::TensorMap<Eigen::Tensor<AccumT, 1, Eigen::RowMajor, Eigen::Index>, Eigen::Unaligned>;

      using Input = Eigen::TensorMap<Eigen::Tensor<const InputT, 1, Eigen::RowMajor, Eigen::Index>,
                                     Eigen::Unaligned>;

      const auto compute = [inner_dim, outer_block_size, buffer_data, input_data,
                            outer_dim](Eigen::Index start, Eigen::Index limit) -> void {
        Eigen::Index outer_dim_start = start * outer_block_size;
        Eigen::Index outer_dim_limit = limit * outer_block_size;
        outer_dim_limit = std::min(outer_dim, outer_dim_limit);

        Buffer buf(buffer_data + start * inner_dim, inner_dim);
        for (Eigen::Index i = outer_dim_start; i < outer_dim_limit; ++i)
        {
          auto in = Input(input_data + i * inner_dim, inner_dim);
          auto cast = in.template cast<AccumT>();
          buf =
            Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf), const decltype(cast)>(
              buf, cast);
        }
      };

      // Compute cost of reducing a single block.
      const Eigen::Index compute_size = outer_block_size * inner_dim;
      const Eigen::Index compute_input_bytes = compute_size * sizeof(InputT);
      const Eigen::TensorOpCost cost(compute_input_bytes,
                                     0, // We'll be mostly writing to L1, assume store cost is 0
                                     compute_size *
                                       Eigen::internal::functor_traits<BinaryFunctor>::Cost);

      device.parallelFor(num_blocks, cost, compute);

      // Aggregate partial results from temporary buffer into first block.
      auto buf0 = Buffer(buffer_data, inner_dim);
      // Just sum the buffer up, as inner dimensions is not large in this case.
      for (int i = 1; i < num_blocks; ++i)
      {
        auto buf = Buffer(buffer_data + i * inner_dim, inner_dim);
        buf0 = Eigen::TensorCwiseBinaryOp<BinaryFunctor, const decltype(buf0), const decltype(buf)>(
          buf0, buf);
      }
      // Write final result to the output.
      output->template flat<OutputT>() = buf0.template cast<OutputT>().reshape(output_dims);
    }
  }
};

void biasReductionHelper(float *input_backprop_buffer, const Shape &input_backprop_shape,
                         float *bias_grad_buffer, const Shape &bias_grad_shape)
{
  assert(input_backprop_buffer);
  assert(bias_grad_buffer);

  const nnfw::cker::functor::ReduceOuterDimensions<Eigen::ThreadPoolDevice, float, float, float,
                                                   Eigen::internal::scalar_sum_op<float>>
    redux;

  const Tensor input_backprop_t{input_backprop_shape, static_cast<void *>(input_backprop_buffer)};

  Tensor bias_grad_t{bias_grad_shape, bias_grad_buffer};

  int outer = 1;
  for (int i = 0; i < input_backprop_shape.DimensionsCount() - 1; ++i)
    outer *= input_backprop_shape.Dims(i);
  int inner = input_backprop_shape.Dims(input_backprop_shape.DimensionsCount() - 1);

  redux(*eigen_support::GetThreadPoolDevice(), Eigen::DSizes<Eigen::Index, 2>{outer, inner},
        input_backprop_t, &bias_grad_t);
}

} // namespace functor
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EIGEN_REDUX_FUNCTOR_H__
