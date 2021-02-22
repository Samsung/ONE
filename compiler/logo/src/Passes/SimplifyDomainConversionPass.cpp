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

#include <logo/SimplifyDomainConversionPass.h>

#include <loco/IR/Algorithm.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNode.h>

#include <memory>
#include <set>
#include <vector>
#include <cassert>

namespace
{

using namespace loco;

// TODO Move this helper into loco
bool equal(const Permutation<Domain::Feature> *lhs, const Permutation<Domain::Feature> *rhs)
{
  for (const auto &axis :
       {FeatureAxis::Count, FeatureAxis::Depth, FeatureAxis::Height, FeatureAxis::Width})
  {
    if (lhs->axis(axis) != rhs->axis(axis))
    {
      return false;
    }
  }
  return true;
}

bool equal(const Permutation<Domain::Filter> *lhs, const Permutation<Domain::Filter> *rhs)
{
  for (const auto &axis :
       {FilterAxis::Count, FilterAxis::Depth, FilterAxis::Height, FilterAxis::Width})
  {
    if (lhs->axis(axis) != rhs->axis(axis))
    {
      return false;
    }
  }
  return true;
}

bool equal(const Permutation<Domain::DepthwiseFilter> *lhs,
           const Permutation<Domain::DepthwiseFilter> *rhs)
{
  for (const auto &axis : {DepthwiseFilterAxis::Depth, DepthwiseFilterAxis::Multiplier,
                           DepthwiseFilterAxis::Height, DepthwiseFilterAxis::Width})
  {
    if (lhs->axis(axis) != rhs->axis(axis))
    {
      return false;
    }
  }
  return true;
}

bool equal(const Permutation<Domain::Matrix> *lhs, const Permutation<Domain::Matrix> *rhs)
{
  for (const auto &axis : {MatrixAxis::Height, MatrixAxis::Width})
  {
    if (lhs->axis(axis) != rhs->axis(axis))
    {
      return false;
    }
  }
  return true;
}

void set_input_null(loco::Node *node)
{
  if (auto casted = dynamic_cast<loco::FeatureEncode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::FeatureDecode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::BiasDecode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::FilterEncode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::FilterDecode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::DepthwiseFilterEncode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::DepthwiseFilterDecode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::MatrixEncode *>(node))
    casted->input(nullptr);
  else if (auto casted = dynamic_cast<loco::MatrixDecode *>(node))
    casted->input(nullptr);
  else
    assert(false && "not supported node type");
}

} // namespace

namespace logo
{

bool SimplifyDomainConversionPass::run(loco::Graph *g)
{
  // TODO Introduce and Use "Pattern Match"
  struct Collector final : public loco::CanonicalNodeMutableVisitor<void>
  {
    // Let's find FeatureDecode followed by FeatureEncode
    void visit(loco::FeatureEncode *encode_node) final
    {
      using namespace loco;

      auto encoder = encode_node->encoder();
      assert(encoder != nullptr);

      auto decode_node = dynamic_cast<loco::FeatureDecode *>(encode_node->input());
      if (decode_node == nullptr)
      {
        return;
      }
      assert(decode_node->input() != nullptr);

      auto decoder = decode_node->decoder();
      assert(decoder != nullptr);

      // NOTE Work only for permuting codec
      auto perm_decoder = dynamic_cast<const PermutingDecoder<Domain::Feature> *>(decoder);
      auto perm_encoder = dynamic_cast<const PermutingEncoder<Domain::Feature> *>(encoder);

      if (perm_encoder == nullptr || perm_decoder == nullptr)
      {
        return;
      }

      if (equal(perm_encoder->perm(), perm_decoder->perm()))
      {
        forwardCandidates.insert({encode_node, decode_node->input()});
      }
    }

    // Let's find `FeatureEncode -- FeatureDecode` pattern
    void visit(loco::FeatureDecode *decode_node) final
    {
      using namespace loco;

      auto encode_node = dynamic_cast<loco::FeatureEncode *>(decode_node->input());
      if (encode_node == nullptr)
      {
        return;
      }
      assert(encode_node->input() != nullptr);

      auto encoder = encode_node->encoder();
      assert(encoder != nullptr);

      auto decoder = decode_node->decoder();
      assert(decoder != nullptr);

      // NOTE Work only for permuting codec
      auto perm_decoder = dynamic_cast<const PermutingDecoder<Domain::Feature> *>(decoder);
      auto perm_encoder = dynamic_cast<const PermutingEncoder<Domain::Feature> *>(encoder);

      if (perm_encoder == nullptr || perm_decoder == nullptr)
      {
        return;
      }

      if (equal(perm_encoder->perm(), perm_decoder->perm()))
      {
        forwardCandidates.insert({decode_node, encode_node->input()});
      }
    }

    // Let's find `FilterEncode -- FilterDecode` pattern
    void visit(loco::FilterDecode *decode_node) final
    {
      using namespace loco;

      auto encode_node = dynamic_cast<loco::FilterEncode *>(decode_node->input());
      if (encode_node == nullptr)
      {
        return;
      }
      assert(encode_node->input() != nullptr);

      auto encoder = encode_node->encoder();
      assert(encoder != nullptr);

      auto decoder = decode_node->decoder();
      assert(decoder != nullptr);

      // NOTE Work only for permuting codec
      auto perm_decoder = dynamic_cast<const PermutingDecoder<Domain::Filter> *>(decoder);
      auto perm_encoder = dynamic_cast<const PermutingEncoder<Domain::Filter> *>(encoder);

      if (perm_encoder == nullptr || perm_decoder == nullptr)
      {
        return;
      }

      if (equal(perm_encoder->perm(), perm_decoder->perm()))
      {
        forwardCandidates.insert({decode_node, encode_node->input()});
      }
      else
      {
        std::vector<loco::TensorAxis> perm_vec;
        perm_vec.resize(4);

        auto enc_perm = perm_encoder->perm();
        auto dec_perm = perm_decoder->perm();

        for (const auto &axis :
             {FilterAxis::Count, FilterAxis::Height, FilterAxis::Width, FilterAxis::Depth})
        {
          auto from = enc_perm->axis(axis);
          auto to = dec_perm->axis(axis);
          perm_vec[to] = from;
        }

        transposeCandidates.insert(
          std::make_unique<TransposeCtx>(encode_node, decode_node, encode_node->input(), perm_vec));
      }
    }

    // Let's find `BiasEncode -- BiasDecode` pattern
    void visit(loco::BiasDecode *decode_node) final
    {
      if (auto encode_node = dynamic_cast<loco::BiasEncode *>(decode_node->input()))
      {
        assert(encode_node->input() != nullptr);
        forwardCandidates.insert({decode_node, encode_node->input()});
      }
    }

    // Let's find `DepthwiseFilterEncode -- DepthwiseFilterDecode` pattern
    void visit(loco::DepthwiseFilterDecode *decode_node) final
    {
      using namespace loco;

      auto encode_node = dynamic_cast<loco::DepthwiseFilterEncode *>(decode_node->input());
      if (encode_node == nullptr)
      {
        return;
      }
      assert(encode_node->input() != nullptr);

      auto encoder = encode_node->encoder();
      assert(encoder != nullptr);

      auto decoder = decode_node->decoder();
      assert(decoder != nullptr);

      // NOTE Work only for permuting codec
      auto perm_decoder = dynamic_cast<const PermutingDecoder<Domain::DepthwiseFilter> *>(decoder);
      auto perm_encoder = dynamic_cast<const PermutingEncoder<Domain::DepthwiseFilter> *>(encoder);

      if (perm_encoder == nullptr || perm_decoder == nullptr)
      {
        return;
      }

      if (equal(perm_encoder->perm(), perm_decoder->perm()))
      {
        forwardCandidates.insert({decode_node, encode_node->input()});
      }
      else
      {
        std::vector<TensorAxis> perm_vec;
        perm_vec.resize(4);

        auto enc_perm = perm_encoder->perm();
        auto dec_perm = perm_decoder->perm();

        for (const auto &axis : {DepthwiseFilterAxis::Depth, DepthwiseFilterAxis::Height,
                                 DepthwiseFilterAxis::Width, DepthwiseFilterAxis::Multiplier})
        {
          auto from = enc_perm->axis(axis);
          auto to = dec_perm->axis(axis);
          perm_vec[to] = from;
        }

        transposeCandidates.insert(
          std::make_unique<TransposeCtx>(encode_node, decode_node, encode_node->input(), perm_vec));
      }
    }

    // Let's find MatrixDecode followed by MatrixEncode
    void visit(loco::MatrixEncode *encode_node) final
    {
      using namespace loco;

      auto encoder = encode_node->encoder();
      assert(encoder != nullptr);

      auto decode_node = dynamic_cast<loco::MatrixDecode *>(encode_node->input());
      if (decode_node == nullptr)
      {
        return;
      }
      assert(decode_node->input() != nullptr);

      auto decoder = decode_node->decoder();
      assert(decoder != nullptr);

      // NOTE Work only for permuting codec
      auto perm_decoder = dynamic_cast<const PermutingDecoder<Domain::Matrix> *>(decoder);
      auto perm_encoder = dynamic_cast<const PermutingEncoder<Domain::Matrix> *>(encoder);

      if (perm_encoder == nullptr || perm_decoder == nullptr)
      {
        return;
      }

      if (equal(perm_encoder->perm(), perm_decoder->perm()))
      {
        forwardCandidates.insert({encode_node, decode_node->input()});
      }
    }

    // Let's find MatrixEncode followed by MatrixDecode
    void visit(loco::MatrixDecode *decode_node) final
    {
      using namespace loco;

      auto encode_node = dynamic_cast<loco::MatrixEncode *>(decode_node->input());
      if (encode_node == nullptr)
      {
        return;
      }
      assert(encode_node->input() != nullptr);

      auto encoder = encode_node->encoder();
      assert(encoder != nullptr);

      auto decoder = decode_node->decoder();
      assert(decoder != nullptr);

      // NOTE Work only for permuting codec
      auto perm_decoder = dynamic_cast<const PermutingDecoder<Domain::Matrix> *>(decoder);
      auto perm_encoder = dynamic_cast<const PermutingEncoder<Domain::Matrix> *>(encoder);

      if (perm_encoder == nullptr || perm_decoder == nullptr)
      {
        return;
      }

      if (equal(perm_encoder->perm(), perm_decoder->perm()))
      {
        forwardCandidates.insert({decode_node, encode_node->input()});
      }
      else
      {
        std::vector<loco::TensorAxis> perm_vec;
        perm_vec.resize(2);

        auto enc_perm = perm_encoder->perm();
        auto dec_perm = perm_decoder->perm();

        for (const auto &axis : {MatrixAxis::Height, MatrixAxis::Width})
        {
          auto from = enc_perm->axis(axis);
          auto to = dec_perm->axis(axis);
          perm_vec[to] = from;
        }

        transposeCandidates.insert(
          std::make_unique<TransposeCtx>(encode_node, decode_node, encode_node->input(), perm_vec));
      }
    }

    void visit(loco::Node *) final { return; }

    using SimplifyingInfo = std::pair<loco::Node * /* end node of subgraph that will be replaced*/,
                                      loco::Node * /* input of subgraph */>;
    std::set<SimplifyingInfo> forwardCandidates;

    struct TransposeCtx
    {
      loco::Node *first_node;                 // starting node of subgraph that will be replaced
      loco::Node *last_node;                  // end node of subgraph that will be replaced
      loco::Node *input_node;                 // input of subgraph
      std::vector<loco::TensorAxis> perm_vec; // perm vector for transpose

      TransposeCtx(loco::Node *first, loco::Node *last, loco::Node *input,
                   std::vector<loco::TensorAxis> perm)
        : first_node(first), last_node(last), input_node(input), perm_vec(perm)
      { /* empty */
      }
    };

    std::set<std::unique_ptr<TransposeCtx>> transposeCandidates;
  };

  Collector collector;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (node->dialect() == loco::CanonicalDialect::get())
    {
      auto canonical_node = loco::must_cast<loco::CanonicalNode *>(node);
      canonical_node->accept(&collector);
    }
  }

  for (auto p : collector.forwardCandidates)
  {
    auto forward_node = g->nodes()->create<loco::Forward>();
    forward_node->input(p.second);
    replace(p.first).with(forward_node);
    set_input_null(p.first);
  }

  for (auto &ctx : collector.transposeCandidates)
  {
    auto transpose_node = g->nodes()->create<loco::TensorTranspose>();
    {
      transpose_node->perm()->size(ctx->perm_vec.size());

      for (loco::TensorAxis axis = 0; axis < ctx->perm_vec.size(); axis++)
        transpose_node->perm()->axis(axis) = ctx->perm_vec[axis];
    }

    transpose_node->input(ctx->input_node);
    replace(ctx->last_node).with(transpose_node);
    set_input_null(ctx->first_node);
  }

  return (collector.forwardCandidates.size() > 0 or collector.transposeCandidates.size() > 0);
}

} // namespace logo
