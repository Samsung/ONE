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

#ifndef __ENCO_COEX_IR_H__
#define __ENCO_COEX_IR_H__

#include <coco/IR.h>

/**
 * @brief 2D Convolution through Andoird NN API
 *
 * TODO Support FusedActivation
 */
class ANNConv2D : public coco::Instr, public coco::Object::Producer, public coco::Object::Consumer
{
public:
  ANNConv2D() : _ofm{this}, _ifm{this}, _ker{this}, _bias{this}
  {
    // DO NOTHING
  }

public:
  coco::Instr *loc(void) override { return this; }

public:
  coco::Object *ofm(void) const { return _ofm.value(); }
  void ofm(coco::Object *o) { _ofm.value(o); }

  coco::Object *ifm(void) const { return _ifm.value(); }
  void ifm(coco::Object *o) { _ifm.value(o); }

  coco::Object *ker(void) const { return _ker.value(); }
  void ker(coco::Object *o) { _ker.value(o); }

  /**
   * Currently, this "bias" is a Feature object with channel-wise layout
   *
   * NOTE This design is subject to change
   */
  coco::Object *bias(void) const { return _bias.value(); }
  void bias(coco::Object *o) { _bias.value(o); }

public:
  coco::Padding2D *pad(void) { return &_pad; }
  const coco::Padding2D *pad(void) const { return &_pad; }

  coco::Stride2D *stride(void) { return &_stride; }
  const coco::Stride2D *stride(void) const { return &_stride; }

private:
  coco::Def _ofm;

  coco::Use _ifm;
  coco::Use _ker;
  coco::Use _bias;

private:
  coco::Padding2D _pad;
  coco::Stride2D _stride;
};

/**
 * @brief Concatenate feature maps along "depth" dimension through Andoird NN API
 */
class ANNDepthConcatF : public coco::Instr,
                        public coco::Object::Producer,
                        public coco::Object::Consumer
{
public:
  ANNDepthConcatF() : _out{this}, _fst{this}, _snd{this}
  {
    // DO NOTHING
  }

public:
  coco::Instr *loc(void) override { return this; }

public:
  coco::Object *out(void) const { return _out.value(); }
  void out(coco::Object *o) { _out.value(o); }

  coco::Object *fst(void) const { return _fst.value(); }
  void fst(coco::Object *o) { _fst.value(o); }

  coco::Object *snd(void) const { return _snd.value(); }
  void snd(coco::Object *o) { _snd.value(o); }

private:
  coco::Def _out;

  // TODO Support variadic-length inputs
  coco::Use _fst;
  coco::Use _snd;
};

#endif // __ENCO_COEX_IR_H__
