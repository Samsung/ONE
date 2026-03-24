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

#ifndef __COCO_IR_OPS_H__
#define __COCO_IR_OPS_H__

#include "coco/IR/Op.h"
#include "coco/IR/Object.h"
#include "coco/IR/KernelObject.h"

#include "coco/IR/Use.h"
#include "coco/IR/Part.h"

#include "coco/IR/Padding2D.h"
#include "coco/IR/Stride2D.h"
#include "coco/IR/Window2D.h"

namespace coco
{

/**
 * @brief Load an Object
 */
class Load final : public Op, public Object::Consumer
{
public:
  explicit Load();

public:
  Load(const Load &) = delete;
  Load(Load &&) = delete;

public:
  uint32_t arity(void) const final;
  Op *arg(uint32_t n) const final;

  std::set<Object *> uses(void) const override;

public:
  Load *asLoad(void) override { return this; }
  const Load *asLoad(void) const override { return this; }

public:
  Instr *loc(void) override { return parent(); }

public:
  void object(Object *o) { _obj.value(o); }
  Object *object(void) const { return _obj.value(); }

private:
  Use _obj;
};

/**
 * @brief 2D Convolution over 3D Feature Map with 4D kernel
 *
 * NOTE IFM and OFM are implicit. Only 4D kernel is explicit in this class
 * TODO Decide source code layout policy and extract this class if necessary
 */
class Conv2D : public Op, public Object::Consumer
{
public:
  explicit Conv2D();

public:
  uint32_t arity(void) const final;
  Op *arg(uint32_t n) const final;

  std::set<Object *> uses(void) const override;

public:
  Conv2D *asConv2D(void) override { return this; }
  const Conv2D *asConv2D(void) const override { return this; }

public:
  Instr *loc(void) override { return parent(); }

private:
  Use _ker;

public:
  Op *arg(void) const { return _arg.child(); }
  void arg(Op *arg) { _arg.child(arg); }

public:
  KernelObject *ker(void) const;
  void ker(KernelObject *ker);

public:
  /**
   * @brief Divide an input and kernel (= convolution filter) into G independent groups
   *
   * Given an input of shape(Ic, Ih, Iw), a kernel of shape(Kn, Kc, Kh, Kw), and group G,
   * Conv2D is identical to G independent convolutions over G inputs of shape(Ic / G, Ih, Iw)
   * and a kernel of shape(Kn / G, Kc, Kh, Kw) followed by concatenation.
   *
   * REQUIRED
   * - "Ic" SHOULD BE a multiple of "G"
   * - "Kc" SHOULD BE identical to "Ic /G"
   *
   * NOTE Depthwise convolution is a special case of group convolution where Ic == G.
   */
  uint32_t group(void) const { return _group; }
  void group(uint32_t g) { _group = g; }

public:
  Padding2D *pad(void) { return &_pad; }
  const Padding2D *pad(void) const { return &_pad; }

public:
  Stride2D *stride(void) { return &_stride; }
  const Stride2D *stride(void) const { return &_stride; }

private:
  uint32_t _group = 1;

  Padding2D _pad;
  Stride2D _stride;

private:
  /// @brief Link to an argument of Conv2D operation (= IFM)
  Part _arg;
};

/**
 * @brief 2D Max Pooling
 */
class MaxPool2D final : public UnaryOp
{
public:
  explicit MaxPool2D() = default;

public:
  MaxPool2D(const MaxPool2D &) = delete;
  MaxPool2D(MaxPool2D &&) = delete;

public:
  MaxPool2D *asMaxPool2D(void) override { return this; }
  const MaxPool2D *asMaxPool2D(void) const override { return this; }

public:
  Window2D *window(void) { return &_window; }
  const Window2D *window(void) const { return &_window; }

public:
  Stride2D *stride(void) { return &_stride; }
  const Stride2D *stride(void) const { return &_stride; }

public:
  Padding2D *pad(void) { return &_pad; }
  const Padding2D *pad(void) const { return &_pad; }

private:
  Window2D _window;
  Stride2D _stride;
  Padding2D _pad;
};

/**
 * @brief 2D Average Pooling
 */
class AvgPool2D final : public UnaryOp
{
public:
  enum class Divisor
  {
    Unknown,
    // Use the number of elements in each receptive field as a divisor
    Static,
    // Use the number of valid (non-padding) elements in each receptive field as a divisor
    PaddingExcluded
  };

public:
  explicit AvgPool2D() = default;

public:
  AvgPool2D(const AvgPool2D &) = delete;
  AvgPool2D(AvgPool2D &&) = delete;

public:
  AvgPool2D *asAvgPool2D(void) override { return this; }
  const AvgPool2D *asAvgPool2D(void) const override { return this; }

public:
  Divisor divisor(void) const { return _divisor; }
  void divisor(const Divisor &divisor) { _divisor = divisor; }

public:
  Window2D *window(void) { return &_window; }
  const Window2D *window(void) const { return &_window; }

public:
  Padding2D *pad(void) { return &_pad; }
  const Padding2D *pad(void) const { return &_pad; }

public:
  Stride2D *stride(void) { return &_stride; }
  const Stride2D *stride(void) const { return &_stride; }

private:
  Divisor _divisor = Divisor::Unknown;

  Window2D _window;
  Stride2D _stride;
  Padding2D _pad;
};

/**
 * @brief Introduce padding area
 */
class PadF final : public UnaryOp
{
public:
  explicit PadF() = default;

public:
  PadF(const PadF &) = delete;
  PadF(PadF &&) = delete;

public:
  PadF *asPadF(void) override { return this; }
  const PadF *asPadF(void) const override { return this; }

public:
  Padding2D *pad(void) { return &_pad; }
  const Padding2D *pad(void) const { return &_pad; }

private:
  Padding2D _pad;
};

/**
 * @brief Apply ReLU over elements
 */
class ReLU final : public UnaryOp
{
public:
  explicit ReLU() = default;

public:
  ReLU(const ReLU &) = delete;
  ReLU(ReLU &&) = delete;

public:
  ReLU *asReLU(void) override { return this; }
  const ReLU *asReLU(void) const override { return this; }
};

/**
 * @brief Apply ReLU6 over elements
 * @note ReLU6 is subject to change
 */
class ReLU6 final : public UnaryOp
{
public:
  explicit ReLU6() = default;

public:
  ReLU6(const ReLU6 &) = delete;
  ReLU6(ReLU6 &&) = delete;

public:
  ReLU6 *asReLU6(void) override { return this; }
  const ReLU6 *asReLU6(void) const override { return this; }
};

/**
 * @brief Element-wise addition
 *
 * Add(L, R) is valid only when L and R have identical kind/shape/dtype
 */
class Add final : public BinaryOp
{
public:
  explicit Add() = default;

public:
  Add(const Add &) = delete;
  Add(Add &&) = delete;

public:
  Add *asAdd(void) override { return this; }
  const Add *asAdd(void) const override { return this; }
};

/**
 * @brief Element-wise subtraction
 *
 * Sub(L, R) is valid only when L and R have identical kind/shape/dtype
 */
class Sub final : public BinaryOp
{
public:
  explicit Sub() = default;

public:
  Sub(const Sub &) = delete;
  Sub(Sub &&) = delete;

public:
  Sub *asSub(void) override { return this; }
  const Sub *asSub(void) const override { return this; }
};

/**
 * @brief Element-wise multiplication
 *
 * Mul(L, R) is valid only when L and R have identical kind/shape/dtype
 */
class Mul final : public BinaryOp
{
public:
  explicit Mul() = default;

public:
  Mul(const Mul &) = delete;
  Mul(Mul &&) = delete;

public:
  Mul *asMul(void) override { return this; }
  const Mul *asMul(void) const override { return this; }
};

/**
 * @brief Element-wise division
 *
 * Div(L, R) is valid only when L and R have identical kind/shape/dtype
 */
class Div final : public BinaryOp
{
public:
  explicit Div() = default;

public:
  Div(const Div &) = delete;
  Div(Div &&) = delete;

public:
  Div *asDiv(void) override { return this; }
  const Div *asDiv(void) const override { return this; }
};

/**
 * @brief Concatenate two feature maps
 *
 * ConcatF(L, R) requires
 */
class ConcatF final : public BinaryOp
{
public:
  enum class Axis
  {
    Unknown = 0,
    Batch = 1,
    Depth = 2,
    Height = 3,
    Width = 4,
  };

public:
  explicit ConcatF() = default;

public:
  ConcatF(const ConcatF &) = delete;
  ConcatF(ConcatF &&) = delete;

public:
  ConcatF *asConcatF(void) override { return this; }
  const ConcatF *asConcatF(void) const override { return this; }

public:
  const Axis &axis(void) const { return _axis; }
  void axis(const Axis &axis) { _axis = axis; }

private:
  Axis _axis = Axis::Unknown;
};

/**
 * @brief Apply Sqrt over elements
 */
class Sqrt final : public UnaryOp
{
public:
  explicit Sqrt() = default;

public:
  Sqrt(const Sqrt &) = delete;
  Sqrt(Sqrt &&) = delete;

public:
  Sqrt *asSqrt(void) override { return this; }
  const Sqrt *asSqrt(void) const override { return this; }
};

} // namespace coco

#endif // __COCO_IR_OPS_H__
