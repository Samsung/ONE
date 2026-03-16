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

#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include <caffe/proto/caffe.pb.h>

#include <coco/IR.h>
#include <coco/IR/Data.h>

#include <cassert>
#include <map>
#include <string>

namespace caffeimport
{

using LayerName = std::string;
using BlobName = std::string;
// Note: these two maybe evolved to a class
using ShapeContext = std::map<BlobName, nncc::core::ADT::tensor::Shape>;
using StoreContext = std::map<BlobName, coco::Bag *>;

class WeightContext
{
public:
  WeightContext(::caffe::NetParameter *caffemodel) : _caffemodel(caffemodel)
  {
    for (uint32_t n = 0; n < _caffemodel->layer_size(); ++n)
    {
      auto layer = _caffemodel->mutable_layer(n);

      if (layer->has_name())
      {
        _data[layer->name()] = layer;
      }
    }
  }

public:
  int blob_count(const LayerName &name)
  {
    if (_data.find(name) != _data.end())
      return _data.at(name)->blobs_size();

    assert(false);
    return 0;
  }

  ::caffe::BlobProto *blob_get(const LayerName &name, uint32_t n)
  {
    if (_data.find(name) != _data.end())
      return _data.at(name)->mutable_blobs(n);

    assert(false);
    return nullptr;
  };

private:
  ::caffe::NetParameter *_caffemodel;
  std::map<LayerName, ::caffe::LayerParameter *> _data;
};

class GraphBuilderContext
{
public:
  explicit GraphBuilderContext(coco::Module *module, coco::Data *data, coco::Block *block,
                               ShapeContext &shape_ctx, StoreContext &bag_ctx,
                               WeightContext &weight_ctx)
    : _module(module), _data(data), _block(block), _shape_ctx(shape_ctx), _bag_ctx(bag_ctx),
      _weight_ctx(weight_ctx)
  {
    // DO NOTHING
  }

  GraphBuilderContext(const GraphBuilderContext &) = delete;
  GraphBuilderContext(GraphBuilderContext &&) = delete;

public:
  coco::Module *module() { return _module; }
  coco::Data *data() { return _data; }
  coco::Block *block() { return _block; }
  ShapeContext &shape_ctx() { return _shape_ctx; }
  StoreContext &bag_ctx() { return _bag_ctx; }
  WeightContext &weight_ctx() { return _weight_ctx; }

private:
  coco::Module *_module;
  coco::Data *_data;
  coco::Block *_block;
  ShapeContext &_shape_ctx;
  StoreContext &_bag_ctx;
  WeightContext &_weight_ctx;
};

} // namespace caffeimport

#endif // __CONTEXT_H__
