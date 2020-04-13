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

#ifndef __NIKE_ABSOLUTE_EPSILON_EQUAL_H__
#define __NIKE_ABSOLUTE_EPSILON_EQUAL_H__

namespace nike
{

class AbsoluteEpsilonEqualFunctor
{
public:
  friend AbsoluteEpsilonEqualFunctor absolute_epsilon_equal(float);

private:
  AbsoluteEpsilonEqualFunctor(float tolerance) : _tolerance{tolerance}
  {
    // DO NOTHING
  }

public:
  bool operator()(float lhs, float rhs) const;

private:
  float _tolerance;
};

/**
 * @note AbsoluteEpsilonEqualFunctor uses its own rule for NaN values.
 *
 * For example, "NAN == NAN" is false but "absolute_epsilon_equal(0.001f)(NAN, NAN)" is true.
 */
AbsoluteEpsilonEqualFunctor absolute_epsilon_equal(float tolerance);

} // namespace nike

#endif // __NIKE_ABSOLUTE_EPSILON_EQUAL_H__
