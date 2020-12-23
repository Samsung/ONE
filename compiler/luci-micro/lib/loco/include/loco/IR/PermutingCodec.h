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

#ifndef __LOCO_IR_PERMUTING_CODEC_H__
#define __LOCO_IR_PERMUTING_CODEC_H__

#include "loco/IR/Domain.h"

#include "loco/IR/FeatureAxis.h"
#include "loco/IR/FeatureCodec.h"
#include "loco/IR/FilterAxis.h"
#include "loco/IR/FilterCodec.h"
#include "loco/IR/DepthwiseFilterAxis.h"
#include "loco/IR/DepthwiseFilterCodec.h"
#include "loco/IR/MatrixAxis.h"
#include "loco/IR/MatrixCodec.h"
#include "loco/IR/TensorAxis.h"

#include <map>

namespace loco
{

template <Domain D> class Permutation;
template <Domain D> class PermutingEncoder;
template <Domain D> class PermutingDecoder;

/**
 * @brief Mapping between Feature/Tensor Axis
 */
template <> class Permutation<Domain::Feature>
{
public:
  Permutation() = default;

public:
  /**
   * @brief Return whether a tensor axis is specified for a given feature axis
   *
   * This method does not validate the corresponding value.
   */
  bool mapped(const FeatureAxis &axis_f) const;

  /**
   * @brief Get the tensor axis corresponding to a given feature axis
   *
   * This method works correclty only when feature axis is mapped before.
   */
  TensorAxis axis(const FeatureAxis &axis_f) const;

  /**
   * @brief Set the tensor axis corresponding to a given feature axis
   */
  TensorAxis &axis(const FeatureAxis &axis_f);

  TensorAxis operator[](const FeatureAxis &axis_f) const { return axis(axis_f); }
  TensorAxis &operator[](const FeatureAxis &axis_f) { return axis(axis_f); }

private:
  std::map<FeatureAxis, TensorAxis> _map;
};

template <> class PermutingEncoder<Domain::Feature> final : public FeatureEncoder
{
public:
  PermutingEncoder() = default;

public:
  PermutingEncoder(const Permutation<Domain::Feature> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  FeatureShape shape(const TensorShape &tensor_shape) const override;
  TensorIndex value(const FeatureIndex &index) const override;

  std::unique_ptr<FeatureEncoder> clone(void) const override;

public:
  const Permutation<Domain::Feature> *perm(void) const { return &_perm; }
  Permutation<Domain::Feature> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::Feature> &p) { _perm = p; }

private:
  Permutation<Domain::Feature> _perm;
};

template <> class PermutingDecoder<Domain::Feature> final : public FeatureDecoder
{
public:
  PermutingDecoder() = default;

public:
  PermutingDecoder(const Permutation<Domain::Feature> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  TensorShape shape(const FeatureShape &tensor_shape) const override;
  FeatureIndex value(const TensorIndex &index) const override;

  std::unique_ptr<FeatureDecoder> clone(void) const override;

public:
  const Permutation<Domain::Feature> *perm(void) const { return &_perm; }
  Permutation<Domain::Feature> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::Feature> &p) { _perm = p; }

private:
  Permutation<Domain::Feature> _perm;
};

/**
 * @brief Mapping between Filter/Tensor Axis
 */
template <> class Permutation<Domain::Filter>
{
public:
  Permutation() = default;

public:
  /**
   * @brief Return whether a given filter axis has a corresponding tensor axis
   *
   * This method does not validate the corresponding value.
   */
  bool mapped(const FilterAxis &axis_f) const;

  /**
   * @brief Get the tensor axis corresponding to a given filter axis
   *
   * This method works correctly only for mapped filter axes.
   */
  const TensorAxis &axis(const FilterAxis &axis_f) const;

  /**
   * @brief Set the tensor axis corresponding to a given filter axis
   */
  TensorAxis &axis(const FilterAxis &axis_f);

  TensorAxis operator[](const FilterAxis &axis_f) const { return axis(axis_f); }
  TensorAxis &operator[](const FilterAxis &axis_f) { return axis(axis_f); }

private:
  std::map<FilterAxis, TensorAxis> _map;
};

/**
 * @brief Permutation-based Tensor-to-Filter converter
 */
template <> class PermutingEncoder<Domain::Filter> final : public FilterEncoder
{
public:
  PermutingEncoder() = default;

public:
  explicit PermutingEncoder(const Permutation<Domain::Filter> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  FilterShape shape(const TensorShape &tensor_shape) const override;
  TensorIndex value(const FilterIndex &index) const override;

public:
  const Permutation<Domain::Filter> *perm(void) const { return &_perm; }
  Permutation<Domain::Filter> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::Filter> &p) { _perm = p; }

private:
  Permutation<Domain::Filter> _perm;
};

/**
 * @brief Permutation-based Filter-to-Tensor converter
 */
template <> class PermutingDecoder<Domain::Filter> final : public FilterDecoder
{
public:
  PermutingDecoder() = default;

public:
  explicit PermutingDecoder(const Permutation<Domain::Filter> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  TensorShape shape(const FilterShape &tensor_shape) const override;
  FilterIndex value(const TensorIndex &index) const override;

public:
  const Permutation<Domain::Filter> *perm(void) const { return &_perm; }
  Permutation<Domain::Filter> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::Filter> &p) { _perm = p; }

private:
  Permutation<Domain::Filter> _perm;
};

/**
 * @brief Mapping between DepthwiseFilter/Tensor Axis
 */
template <> class Permutation<Domain::DepthwiseFilter>
{
public:
  Permutation() = default;

public:
  /**
   * @brief Return whether a given depthwise filter axis has a corresponding tensor axis
   *
   * This method does not validate the corresponding value.
   */
  bool mapped(const DepthwiseFilterAxis &axis_f) const;

  /**
   * @brief Get the tensor axis corresponding to a given depthwise filter axis
   *
   * This method works correctly only for mapped depthwise filter axes.
   */
  const TensorAxis &axis(const DepthwiseFilterAxis &axis_f) const;

  /**
   * @brief Set the tensor axis corresponding to a given depthwise filter axis
   */
  TensorAxis &axis(const DepthwiseFilterAxis &axis_f);

  TensorAxis operator[](const DepthwiseFilterAxis &axis_f) const { return axis(axis_f); }
  TensorAxis &operator[](const DepthwiseFilterAxis &axis_f) { return axis(axis_f); }

private:
  std::map<DepthwiseFilterAxis, TensorAxis> _map;
};

/**
 * @brief Permutation-based Tensor-to-DepthwiseFilter converter
 */
template <> class PermutingEncoder<Domain::DepthwiseFilter> final : public DepthwiseFilterEncoder
{
public:
  PermutingEncoder() = default;

public:
  PermutingEncoder(const Permutation<Domain::DepthwiseFilter> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  DepthwiseFilterShape shape(const TensorShape &tensor_shape) const override;
  TensorIndex value(const DepthwiseFilterIndex &index) const override;

public:
  const Permutation<Domain::DepthwiseFilter> *perm(void) const { return &_perm; }
  Permutation<Domain::DepthwiseFilter> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::DepthwiseFilter> &p) { _perm = p; }

private:
  Permutation<Domain::DepthwiseFilter> _perm;
};

/**
 * @brief Permutation-based DepthwiseFilter-to-Tensor converter
 */
template <> class PermutingDecoder<Domain::DepthwiseFilter> final : public DepthwiseFilterDecoder
{
public:
  PermutingDecoder() = default;

public:
  PermutingDecoder(const Permutation<Domain::DepthwiseFilter> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  TensorShape shape(const DepthwiseFilterShape &shape) const override;
  DepthwiseFilterIndex value(const TensorIndex &index) const override;

public:
  const Permutation<Domain::DepthwiseFilter> *perm(void) const { return &_perm; }
  Permutation<Domain::DepthwiseFilter> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::DepthwiseFilter> &p) { _perm = p; }

private:
  Permutation<Domain::DepthwiseFilter> _perm;
};

/**
 * @brief Mapping between Matrix/Tensor Axis
 */
template <> class Permutation<Domain::Matrix>
{
public:
  Permutation() = default;

public:
  /**
   * @brief Return whether a given matrix axis has a corresponding tensor axis
   *
   * This method does not validate the corresponding value.
   */
  bool mapped(const MatrixAxis &axis_f) const;

  /**
   * @brief Get the tensor axis corresponding to a given matrix axis
   *
   * This method works correctly only for mapped matrix axes.
   */
  TensorAxis axis(const MatrixAxis &axis_f) const;

  /**
   * @brief Set the tensor axis corresponding to a given matrix axis
   */
  TensorAxis &axis(const MatrixAxis &axis_f);

  TensorAxis operator[](const MatrixAxis &axis_f) const { return axis(axis_f); }
  TensorAxis &operator[](const MatrixAxis &axis_f) { return axis(axis_f); }

private:
  std::map<MatrixAxis, TensorAxis> _map;
};

/**
 * @brief Permutation-based Tensor-to-Matrix converter
 */
template <> class PermutingEncoder<Domain::Matrix> final : public MatrixEncoder
{
public:
  PermutingEncoder() = default;

public:
  PermutingEncoder(const Permutation<Domain::Matrix> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  MatrixShape shape(const TensorShape &tensor_shape) const override;
  TensorIndex value(const MatrixIndex &index) const override;

public:
  const Permutation<Domain::Matrix> *perm(void) const { return &_perm; }
  Permutation<Domain::Matrix> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::Matrix> &p) { _perm = p; }

private:
  Permutation<Domain::Matrix> _perm;
};

/**
 * @brief Permutation-based Matrix-to-Tensor converter
 */
template <> class PermutingDecoder<Domain::Matrix> final : public MatrixDecoder
{
public:
  PermutingDecoder() = default;

public:
  PermutingDecoder(const Permutation<Domain::Matrix> &perm) : _perm{perm}
  {
    // DO NOTHING
  }

public:
  bool valid(void) const;

public:
  TensorShape shape(const MatrixShape &tensor_shape) const override;
  MatrixIndex value(const TensorIndex &index) const override;

public:
  const Permutation<Domain::Matrix> *perm(void) const { return &_perm; }
  Permutation<Domain::Matrix> *perm(void) { return &_perm; }
  void perm(const Permutation<Domain::Matrix> &p) { _perm = p; }

private:
  Permutation<Domain::Matrix> _perm;
};

} // namespace loco

#endif // __LOCO_IR_PERMUTING_CODEC_H__
