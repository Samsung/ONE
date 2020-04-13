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

/*
 * Copyright (c) 2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef __IO_ACCESSOR_H__
#define __IO_ACCESSOR_H__

#include <arm_compute/graph/ITensorAccessor.h>

class InputAccessor : public arm_compute::graph::ITensorAccessor
{
public:
  InputAccessor(bool inc) : _inc(inc) { _test_input = 1.0; }
  InputAccessor(InputAccessor &&) = default;

  // Inherited methods overriden:
  bool access_tensor(arm_compute::ITensor &tensor) override;

private:
  bool _inc;
  float _test_input;
};

class OutputAccessor : public arm_compute::graph::ITensorAccessor
{
public:
  OutputAccessor() = default;
  OutputAccessor(OutputAccessor &&) = default;

  // Inherited methods overriden:
  bool access_tensor(arm_compute::ITensor &tensor) override;
};

class WeightAccessor : public arm_compute::graph::ITensorAccessor
{
public:
  WeightAccessor(bool inc) : _inc(inc) { _test_weight = 1.0; }
  WeightAccessor(WeightAccessor &&) = default;

  // Inherited methods overriden:
  bool access_tensor(arm_compute::ITensor &tensor) override;

private:
  bool _inc;
  float _test_weight;
};

class BiasAccessor : public arm_compute::graph::ITensorAccessor
{
public:
  BiasAccessor() = default;
  BiasAccessor(BiasAccessor &&) = default;

  // Inherited methods overriden:
  bool access_tensor(arm_compute::ITensor &tensor) override;
};

#endif // __IO_ACCESSOR_H__
