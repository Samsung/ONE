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

#ifndef __LOCOEX_COPATTRTYPES_H__
#define __LOCOEX_COPATTRTYPES_H__

#include <stdexcept>

namespace locoex
{

/**
 * @brief Tensorflow attribute type
 *        Refer to https://www.tensorflow.org/guide/extend/op#attr_types
 */
enum class COpAttrType
{
  Int,
  Float,
  // TODO Support more attr types such as String, Bool, DataType, Tensor, Shape, List
};

/**
 * @brief Struct that holds attr type
 */
struct COpAttrData
{
protected:
  COpAttrData(COpAttrType attr_type) : _type(attr_type) {}

public:
  virtual ~COpAttrData() = default;

public:
  COpAttrType type() const { return _type; }
  void type(COpAttrType attr_type) { _type = attr_type; }

private:
  COpAttrType _type;
};

/**
 * @brief Struct that holds attr data of int type
 */
struct COpAttrInt final : public COpAttrData
{
public:
  COpAttrInt(int tf_val) : COpAttrData(COpAttrType::Int) { _val = tf_val; }

  int val() const { return _val; }
  void val(int val) { _val = val; }

private:
  int _val;
};

/**
 * @brief Struct that holds attr data of float type
 */
struct COpAttrFloat final : public COpAttrData
{
public:
  COpAttrFloat(float tf_val) : COpAttrData(COpAttrType::Float) { _val = tf_val; }

  float val() const { return _val; }
  void val(float val) { _val = val; }

private:
  float _val;
};

template <COpAttrType AT> struct AttrTypeTrait;

template <> struct AttrTypeTrait<COpAttrType::Float>
{
  using Type = COpAttrFloat;
};

template <> struct AttrTypeTrait<COpAttrType::Int>
{
  using Type = COpAttrInt;
};

// TODO support more attr types

} // namespace locoex

#endif // __LOCOEX_COPATTRTYPES_H__
