/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __GRAPH_BLOCK_H__
#define __GRAPH_BLOCK_H__

#include <loco.h>
#include <loco/Service/ShapeInference.h>

#include <oops/InternalExn.h>

#include <functional>

namespace exo
{

/// @brief feature layout of TFLITE file
enum class FeatureLayout
{
  NHWC,
};

/// @brief Creates a loco::FeatureEncode with T layout (NHWC for tflite) and add it to graph.
template <FeatureLayout T> loco::FeatureEncode *make_feature_encode(loco::Node *input_for_encode);

/// @brief Creates a loco::FeatureDecode with T layout (NHWC for tflite) and add it to graph.
template <FeatureLayout T> loco::FeatureDecode *make_feature_decode(loco::Node *input_for_decode);

enum class FilterLayout
{
  OHWI, // a.k.a., NHWC, Tensorflow Lite uses this layout for filter
  HWIO, // a.k.a., HWCN, Tensorflow uses this layout for filter
};

/// @brief Create a loco::FilterEncode of given layout
template <FilterLayout T> loco::FilterEncode *make_filter_encode(loco::Node *input_for_encode);

/// @brief Create a loco::FilterDecode of given layout
template <FilterLayout T> loco::FilterDecode *make_filter_decode(loco::Node *input_for_decode);

enum class DepthwiseFilterLayout
{
  HWCM,
};

/// @brief Create a loco::DepthwiseFilterDecode of given layout
template <DepthwiseFilterLayout T>
loco::DepthwiseFilterDecode *make_dw_filter_decode(loco::Node *input_for_decode);

enum class MatrixLayout
{
  HW,
  WH
};

/// @brief Create a loco::MatrixEncode of given layout
template <MatrixLayout T> loco::MatrixEncode *make_matrix_encode(loco::Node *input_for_encode);

/// @brief Create a loco::MatrixDecode of given layout
template <MatrixLayout T> loco::MatrixDecode *make_matrix_decode(loco::Node *input_for_decode);

} // namespace exo

//
// DomainConverter
//

/**
 * Some canonical nodes can have input of various loco::Domain, e.g., loco::Domain::Tensor,
 * loco::Domain::Feature, etc. However, TFL node accepts only loco::Domain::Tensor.
 * So, When converting such canonical node to TFL node and input(s) of a canonical node are not
 * loco::Domain::Tensor, additional nodes need to be inserted.
 *
 * The following two classes helps this insertion.
 *
 * For example, in case of loco::Relu conversion,
 *
 * Before:
 *
 *    A (output: feature) -- loco::ReLU --- B (input:feature)
 *
 * After:
 *
 *    A -- loco::FeatureDecode -- locoex::TFLRelu -- loco::FeatureEncode --- B
 *
 *                  loco::ReLU (dead node)
 */

namespace exo
{

/**
 * @brief Handles input(s) while converting a canonical node to TFL node(s).
 *        This class informs DomainConverter how to handle inputs of a specific canonical node.
 */
template <class CanonicalT, class TFLT> class InputHandler
{
public:
  /**
   * @brief Assign origin's inputs to replacer's inputs.
   *        (This is called when origin belongs in Tensor domain.)
   */
  virtual void handover(CanonicalT *origin, TFLT *replacer) = 0;

  /**
   * @brief Returns the list of inputs that needs to have FeatureDecode as its input.
   *        (This is called when origin belongs in Feature domain.)
   */
  virtual std::vector<loco::Node *> getInputsToConvert(CanonicalT *origin) = 0;

  /// @brief Set the inputs of replacer to new_inputs
  virtual void set(TFLT *replacer, std::vector<loco::Node *> &new_inputs) = 0;

  /// @brief Set the inputs to nullptr
  virtual void nullify(CanonicalT *origin) = 0;
};

/**
 * @brief Class to handle domain conversion while converting a canonical node to TFL node(s)
 */
template <class CanonicalT, class TFLT> class DomainConverter
{
public:
  template <FeatureLayout FeatureLayoutT>
  TFLT *convert(CanonicalT *origin, InputHandler<CanonicalT, TFLT> &input_handler);
};

/**
 * @brief Performs domain conversion
 *
 * 1. if origin belong to loco::Domain::Tensor, and replace origin to a TFL node.
 * 2. if origin belong to loco::Domain::Feature, insert loco::FeatureDecode for input(s) and
 *    insert loco::FeatureEncode for output. Then replace origin to a TFL node.
 *
 * @return new TFL node; nullptr if shape of origin cannot be known
 */
template <class CanonicalT, class TFLT>
template <FeatureLayout FeatureLayoutT>
TFLT *DomainConverter<CanonicalT, TFLT>::convert(CanonicalT *origin,
                                                 InputHandler<CanonicalT, TFLT> &input_handler)
{
  static_assert(FeatureLayoutT == FeatureLayout::NHWC, "Feature layout should be NHWC");

  if (!loco::shape_known(origin))
  {
    return nullptr;
  }

  auto tfl_node = origin->graph()->nodes()->template create<TFLT>();

  // when the input is Tensor, just replace canonical node to TFL node.
  if (loco::shape_get(origin).domain() == loco::Domain::Tensor)
  {
    input_handler.handover(origin, tfl_node);

    loco::replace(origin).with(tfl_node);
    input_handler.nullify(origin);

    return tfl_node;
  }
  else if (loco::shape_get(origin).domain() == loco::Domain::Feature)
  {
    std::vector<loco::Node *> feature_decodes;

    for (auto input : input_handler.getInputsToConvert(origin))
    {
      auto dec = make_feature_decode<FeatureLayoutT>(input);
      feature_decodes.emplace_back(dec);
    }

    input_handler.set(tfl_node, feature_decodes);

    auto enc = make_feature_encode<FeatureLayoutT>(tfl_node);

    loco::replace(origin).with(enc);
    input_handler.nullify(origin);

    return tfl_node;
  }
  else
    INTERNAL_EXN_V("Unsupported loco::Domain", oops::to_uint32(loco::shape_get(origin).domain()));
}

} // namespace exo

#endif //__GRAPH_BLOCK_H__
