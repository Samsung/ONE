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

#include "Host.h"

#include <pp/EnclosedDocument.h>

#include <stdex/Memory.h>

#include <map>
#include <string>

namespace
{

/**
 * @brief Data transfer between flat arrays
 *
 * Transfer(from, into) denotes the following C code:
 *   dst[into] = src[from];
 */
class Transfer
{
public:
  Transfer(uint32_t from, uint32_t into) : _from{from}, _into{into}
  {
    // DO NOTHING
  }

public:
  uint32_t from(void) const { return _from; }
  uint32_t into(void) const { return _into; }

private:
  uint32_t _from;
  uint32_t _into;
};

using TransferSequence = std::vector<Transfer>;

/**
 * @brief Convert Shuffle instruction as a sequence of data transfer
 */
TransferSequence as_transfer_sequence(const coco::Shuffle *shuffle)
{
  TransferSequence seq;

  for (const auto &dst : shuffle->range())
  {
    const auto src = shuffle->at(dst);
    seq.emplace_back(src.value(), dst.value());
  }

  return seq;
}

/**
 * Given a sequence of N data transfers,
 * find_loop tries to compute count, src_step, dst_step that satisfies
 * the following properties:
 *
 * First, N should be a multiple of count.
 *        Below we refer to that multiplier as 'window' (= N / count)
 *
 * Second,
 *   for all n in [0, count),
 *     for all k in [0, window),
 *       from[n * window + k] == from[k] + src_step, and
 *       into[n * window + k] == into[k] + dst_step
 */
bool find_loop(TransferSequence::const_iterator beg, TransferSequence::const_iterator end,
               uint32_t *p_count, uint32_t *p_src_step, uint32_t *p_dst_step)
{
  assert(p_count != nullptr);
  assert(p_src_step != nullptr);
  assert(p_dst_step != nullptr);

  const uint32_t size = end - beg;

  for (uint32_t window = 1; window <= size; ++window)
  {
    if (size % window != 0)
    {
      continue;
    }

    auto src_step_at = [&beg, window](uint32_t n) {
      return (beg + n)->from() - (beg + n - window)->from();
    };

    auto dst_step_at = [&beg, window](uint32_t n) {
      return (beg + n)->into() - (beg + n - window)->into();
    };

    const uint32_t count = size / window;
    const uint32_t src_step = src_step_at(window);
    const uint32_t dst_step = dst_step_at(window);

    bool consistent = true;

    for (uint32_t n = window + 1; n < size; ++n)
    {
      if ((src_step_at(n) != src_step) || (dst_step_at(n) != dst_step))
      {
        consistent = false;
        break;
      }
    }

    if (consistent)
    {
      *p_count = count;
      *p_src_step = src_step;
      *p_dst_step = dst_step;
      return true;
    }
  }

  return false;
}

/**
 * @brief Single transfer loop (a triple of count, source step, detination step)
 */
class TransferLoop
{
public:
  class Step
  {
  public:
    Step(uint32_t src, uint32_t dst) : _src{src}, _dst{dst}
    {
      // DO NOTHING
    }

  public:
    uint32_t src(void) const { return _src; }
    uint32_t dst(void) const { return _dst; }

  private:
    uint32_t _src;
    uint32_t _dst;
  };

public:
  TransferLoop(uint32_t count, uint32_t src_step, uint32_t dst_step)
    : _count{count}, _step{src_step, dst_step}
  {
    // DO NOTHING
  }

public:
  uint32_t count(void) const { return _count; }
  const Step &step(void) const { return _step; }

private:
  uint32_t _count;
  Step _step;
};

/**
 * @brief Nested transfer loops
 */
using TransferNest = std::vector<TransferLoop>;

/**
 * @brief Construct nested transfer loop-nest that correponds to a given Shuffle instruction
 */
TransferNest as_nest(const TransferSequence &seq)
{
  TransferNest nest;

  auto beg = seq.begin();
  auto end = seq.end();

  uint32_t window = end - beg;
  uint32_t count = 0;
  uint32_t src_step = 0;
  uint32_t dst_step = 0;

  while ((window > 1) && find_loop(beg, end, &count, &src_step, &dst_step))
  {
    assert(window % count == 0);

    window /= count;
    end = beg + window;

    nest.emplace_back(count, src_step, dst_step);
  }

  return nest;
};

uint32_t loop_count(const TransferNest &nest)
{
  uint32_t count = 1;

  for (const auto &loop : nest)
  {
    count *= loop.count();
  }

  return count;
};

class InstrPrinter : public coco::Instr::Visitor<pp::LinearDocument>
{
public:
  InstrPrinter(const enco::MemoryContext &mem) : _mem(mem)
  {
    // DO NOTHING
  }

private:
  pp::LinearDocument visit(const coco::Shuffle *shuffle) override
  {
    auto from = shuffle->from();
    auto into = shuffle->into();

    //
    // Analyze 'Shuffle' pattern, and convert it as nested loops
    //
    auto tseq = as_transfer_sequence(shuffle);
    auto nest = as_nest(tseq);
    assert(tseq.size() % loop_count(nest) == 0);
    uint32_t window = tseq.size() / loop_count(nest);

    //
    // Generate loop body
    //
    pp::EnclosedDocument loop_body;

    auto var_at = [](uint32_t lv) { return pp::fmt("_", lv); };

    for (uint32_t lv = 0; lv < nest.size(); ++lv)
    {
      auto var = var_at(lv);

      loop_body.front().append("for (uint32_t ", var, " = 0; ", var, " < ", nest.at(lv).count(),
                               "; ++", var, ") {");
      loop_body.front().indent();

      loop_body.back().append("}");
      loop_body.back().indent();
    }

    std::string src_index = "0";
    std::string dst_index = "0";

    for (uint32_t lv = 0; lv < nest.size(); ++lv)
    {
      src_index += pp::fmt(" + ", nest.at(lv).step().src(), " * ", var_at(lv));
      dst_index += pp::fmt(" + ", nest.at(lv).step().dst(), " * ", var_at(lv));
    }

    for (uint32_t n = 0; n < window; ++n)
    {
      const auto src_base = pp::fmt("reinterpret_cast<const float *>(", _mem.base(from), ")");
      const auto dst_base = pp::fmt("reinterpret_cast<float *>(", _mem.base(into), ")");

      loop_body.front().append(dst_base, "[", dst_index, " + ", tseq.at(n).into(), "] = ", src_base,
                               "[", src_index, " + ", tseq.at(n).from(), "];");
    }

    pp::LinearDocument res;
    res.append(loop_body);
    return res;
  }

private:
  const enco::MemoryContext &_mem;
};

} // namespace

namespace enco
{

std::unique_ptr<pp::MultiLineText> HostBlockCompiler::compile(const coco::Block *blk) const
{
  InstrPrinter prn{_mem};

  auto res = stdex::make_unique<pp::LinearDocument>();

  for (auto ins = blk->instr()->head(); ins; ins = ins->next())
  {
    res->append(ins->accept(prn));
  }

  return std::move(res);
}

} // namespace enco
