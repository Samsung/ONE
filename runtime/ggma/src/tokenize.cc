/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tokenize.h"

#include <cstring>

namespace ggma
{

size_t GGMATokenizer::tokenize(const char *text, size_t text_len, int64_t *tokens,
                               size_t max_tokens, size_t *actual_count) const
{
  // TODO: Implement actual tokenization using SentencePiece
  // For now, return the same hardcoded tokens as the original stub implementation

  if (!text || !tokens || !actual_count || max_tokens == 0)
  {
    if (actual_count)
      *actual_count = 0;
    return 0;
  }

  // Hardcoded tokens for "Lily picked up a flower." (same as original stub)
  // IMPORTANT: Keep the original 32-element array and copy the entire array!
  int64_t tokenized[32] = {
    1, 21075, 7727, 550, 260, 12584, 31843,
  };

  *actual_count = 7;
  memcpy(tokens, tokenized, sizeof(tokenized));

  return *actual_count;
}

} // namespace ggma
