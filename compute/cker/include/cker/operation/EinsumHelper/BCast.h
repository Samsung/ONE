/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EINSUM_HELPER_BCAST_H__
#define __NNFW_CKER_EINSUM_HELPER_BCAST_H__

namespace nnfw
{
namespace cker
{

// BCast is a helper for broadcasting binary tensor operation.
// TensorFlow's broadcasting rule follows that of numpy (See
// http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).
//
// The rule has the following properties:
//
//   1. suffix matching: the rule starts with the right-most
//      dimension, and works towards the left-most dimension. Since
//      TensorFlow is row-major, the right-most dimension (the last
//      element in the shape of a tensor) is the inner-most, a.k.a.
//      the fastest changing, dimension.
//
//   2. Two dimensions are compatible for broadcasting if both are the
//      same or either is 1.
//
// BCast takes the shape of two tensors and computes a few vectors of
// int32 that are useful for the caller to reshape the tensors, apply
// the right broadcasts to them, compute the broadcasted operation,
// and possibly the gradients. In a nutshell, the caller is expected
// to compute the broadcasted operation as following:
//
//   BCast b(x.shape(), y.shape());
//   output = x.reshape(b.x_reshape()).broadcast(b.x_bcast())
//            _op_
//            y.reshape(b.y_reshape()).broadcast(b.y_bcast())
//
// For the gradient computation,
//   grad_x = sum(grad * backprop_x(x, y), grad_x_reduce_idx)
//            .reshape(x.shape())
//   grad_y = sum(grad * backprop_y(x, y), grad_y_reduce_idx)
//            .reshape(y.shape())
// backprop_x and backprop_y are functionals of the binary function "op",
// e.g.,
//   for +, backprop_x(x, y) = backprop_y(x, y) = 1;
//   for *, backprop_x(x, y) =  y, backprop_y(x, y) = x;
//   for /, backprop_x(x, y) = 1/y, backprop_y(x, y) = -x/y^2;
//
// The multiplication in the grad * backprop_x itself is also
// broadcasting following the same rule.
//
// TODO(zhifengc): Adds support for n-ary (n >= 2).
class BCast
{
public:
  // Constructs all helper shapes, following the aforementioned rules.
  //
  // If "fewer_dims_optimization" is set to true (the default), the
  // implementation tries to reduce intermediate dimensions needed to be more
  // efficient.  This is transparent to the caller.
  //
  // If false, all intermediate shapes (except for grad_{x,y}_reduce_idx()) have
  // the same number of dimensions as the larger of the two inputs.
  BCast(const std::vector<int32_t> &sx, const std::vector<int32_t> &sy,
        const bool fewer_dims_optimization = true)
  {
    if (sx == sy && fewer_dims_optimization)
    {
      // Fast path for common case of identical shapes for sx and sy
      int32_t elements = 1;
      const int n = sx.size();
      _output.resize(n);
      for (int i = 0; i < n; i++)
      {
        const int32_t dim = sx[i];
        elements *= dim;
        _output[i] = dim;
      }
      result_.push_back(elements);
      _x_reshape.push_back(elements);
      _y_reshape.push_back(elements);
      // grad_x_reduce_ and grad_y_reduce_ are left as empty
    }
    else
    {
      // Reverse the shape of x and y for convenience.
      // After the reverse, 0-th is the inner-most dimension.
      std::vector<int32_t> x = sx;
      std::vector<int32_t> y = sy;
      Reverse(&x);
      Reverse(&y);

      // 1-extend and align x and y so that they are the same size.
      if (x.size() > y.size())
      {
        y.resize(x.size(), 1);
      }
      else
      {
        x.resize(y.size(), 1);
      }

      // Going through each dimension starting from the inner-most
      // dimension, compares dimension of x and y. They are compatible if
      // they are equal or either is 1.
      enum State
      {
        UNKNOWN,
        SAME,
        X_ONE,
        Y_ONE,
      };
      State prev = UNKNOWN;
      const int32_t n = x.size();
      for (int i = 0; i < n; ++i)
      {
        // Output shape.
        State curr = UNKNOWN;
        const int32_t x_i = x[i]; // i-th dimension of x.
        const int32_t y_i = y[i]; // i-th dimension of y.
        int32_t o_i;              // i-th dimension of the output.
        // Invariant:
        //   o_i = x_i * bx_i = y_i * by_i
        if (x_i == y_i)
        {
          // No broadcast.
          o_i = x_i;
          curr = SAME;
        }
        else if (x_i == 1)
        {
          // x broadcast to y on this dimension.
          o_i = y_i;
          curr = X_ONE;
        }
        else if (y_i == 1)
        {
          // y broadcast to x on this dimension.
          o_i = x_i;
          curr = Y_ONE;
        }
        else
        {
          _valid = false;
          return;
        }
        _output.push_back(o_i);
        // Reshape/broadcast.
        // Invariant:
        //  result[i] == x_reshape[i] * x_bcast[i] == _y_reshape[i] * y_bcast_[i]
        if (curr == SAME && x_i == 1)
        {
          if (!fewer_dims_optimization)
          {
            result_.push_back(o_i);
            _x_reshape.push_back(x_i);
            _y_reshape.push_back(y_i);
          }
          continue;
        }
        else if (fewer_dims_optimization && prev == curr)
        {
          // It is a run of the same cases(no broadcast, x broadcast to y, y
          // broadcast to x). We can reshape the input so that fewer dimensions
          // are involved in the intermediate computation.
          result_.back() *= o_i;
          _x_reshape.back() *= x_i;
          _y_reshape.back() *= y_i;
        }
        else
        {
          result_.push_back(o_i);
          _x_reshape.push_back(x_i);
          _y_reshape.push_back(y_i);
        }
        prev = curr;
      }

      if (result_.empty())
      {
        // Can happen when both x and y are effectively scalar.
        result_.push_back(1);
        _x_reshape.push_back(1);
        _y_reshape.push_back(1);
      }

      // Reverse all vectors since x and y were reversed at very
      // beginning.
      Reverse(&_x_reshape);
      Reverse(&_y_reshape);
      Reverse(&result_);
      Reverse(&_output);
    }
  }
  ~BCast() {}

  // Returns true iff two operands are compatible according to the
  // broadcasting rule.
  bool IsValid() const { return _valid; }

  // If and only if IsValid(), the following fields can be used in
  // implementing a broadcasted binary tensor operation according to
  // the broadcasting rule.
  const std::vector<int32_t> &x_reshape() const { return _x_reshape; }
  const std::vector<int32_t> &y_reshape() const { return _y_reshape; }
  const std::vector<int32_t> &output_shape() const { return _output; }

private:
  void Reverse(std::vector<int32_t> *shape) { std::reverse(shape->begin(), shape->end()); }

private:
  bool _valid = true;
  std::vector<int32_t> _x_reshape;
  std::vector<int32_t> _y_reshape;
  std::vector<int32_t> result_;
  std::vector<int32_t> _output;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EINSUM_HELPER_BCAST_H__
