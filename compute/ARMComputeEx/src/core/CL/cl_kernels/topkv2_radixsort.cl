/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2017 ARM Limited.
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

// reference:
// https://code.google.com/archive/p/ocl-radix-sort/source/default/source
// OpenCL kernel sources for the CLRadixSort class
// the #include does not exist in OpenCL
// Copyright Philippe Helluy, Université de Strasbourg, France, 2011, helluy@math.unistra.fr
// licensed under the GNU Lesser General Public License see http://www.gnu.org/copyleft/lesser.html
// if you find this software usefull you can cite the following work in your reports or articles:
// Philippe HELLUY, A portable implementation of the radix sort algorithm in OpenCL, 2011.
// http://hal.archives-ouvertes.fr/hal-00596730

// Reference for floating point radix sort:
// http://www.codercorner.com/RadixSortRevisited.htm

// compute the histogram for each radix and each virtual processor for the pass
__kernel void radixsort_histogram(__global float *in_key_buf, __global int *d_Histograms,
                                  const int pass, __local int *loc_histo, const int n)
{
  int it = get_local_id(0);  // i local number of the processor
  int ig = get_global_id(0); // global number = i + g I

  int gr = get_group_id(0); // g group number

  int groups = get_num_groups(0);
  int items = get_local_size(0);

  // set the local histograms to zero
  for (int ir = 0; ir < _RADIX; ir++)
  {
    loc_histo[ir * items + it] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // range of keys that are analyzed by the work item
  int size = n / groups / items; // size of the sub-list
  int start = ig * size;         // beginning of the sub-list

  unsigned int key;
  int shortkey, k;

  // compute the index
  // the computation depends on the transposition
  for (int j = 0; j < size; j++)
  {
#ifdef TRANSPOSE
    k = groups * items * j + ig;
#else
    k = j + start;
#endif

    key = *((__global unsigned int *)(in_key_buf + k));

    // extract the group of _BITS bits of the pass
    // the result is in the range 0.._RADIX-1
    shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1));

    // increment the local histogram
    loc_histo[shortkey * items + it]++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // copy the local histogram to the global one
  for (int ir = 0; ir < _RADIX; ir++)
  {
    d_Histograms[items * (ir * groups + gr) + it] = loc_histo[ir * items + it];
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
}

// initial transpose of the list for improving
// coalescent memory access
__kernel void transpose(const __global int *invect, __global int *outvect, const int nbcol,
                        const int nbrow, const __global int *inperm, __global int *outperm,
                        __local int *blockmat, __local int *blockperm, const int tilesize)
{

  int i0 = get_global_id(0) * tilesize; // first row index
  int j = get_global_id(1);             // column index

  int jloc = get_local_id(1); // local column index

  // fill the cache
  for (int iloc = 0; iloc < tilesize; iloc++)
  {
    int k = (i0 + iloc) * nbcol + j; // position in the matrix
    blockmat[iloc * tilesize + jloc] = invect[k];
#ifdef PERMUT
    blockperm[iloc * tilesize + jloc] = inperm[k];
#endif
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // first row index in the transpose
  int j0 = get_group_id(1) * tilesize;

  // put the cache at the good place
  for (int iloc = 0; iloc < tilesize; iloc++)
  {
    int kt = (j0 + iloc) * nbrow + i0 + jloc; // position in the transpose
    outvect[kt] = blockmat[jloc * tilesize + iloc];
#ifdef PERMUT
    outperm[kt] = blockperm[jloc * tilesize + iloc];
#endif
  }
}

// each virtual processor reorders its data using the scanned histogram
__kernel void radixsort_reorder(__global float *in_key, __global float *out_key,
                                __global int *d_Histograms, const int pass,
                                __global int *indices_in, __global int *indices_out,
                                __local int *loc_histo, const int n)
{

  int it = get_local_id(0);
  int ig = get_global_id(0);

  int gr = get_group_id(0);
  int groups = get_num_groups(0);
  int items = get_local_size(0);

  int start = ig * (n / groups / items);
  int size = n / groups / items;

  // take the histogram in the cache
  for (int ir = 0; ir < _RADIX; ir++)
  {
    loc_histo[ir * items + it] = d_Histograms[items * (ir * groups + gr) + it];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  int newpos, shortkey, k, newpost;
  unsigned int key;

  for (int j = 0; j < size; j++)
  {
#ifdef TRANSPOSE
    k = groups * items * j + ig;
#else
    k = j + start;
#endif
    float org_value = in_key[k];
    key = *(__global unsigned int *)(in_key + k);
    shortkey = ((key >> (pass * _BITS)) & (_RADIX - 1));

    newpos = loc_histo[shortkey * items + it];

#ifdef TRANSPOSE
    int ignew, jnew;
    ignew = newpos / (n / groups / items);
    jnew = newpos % (n / groups / items);
    newpost = jnew * (groups * items) + ignew;
#else
    newpost = newpos;
#endif

    // d_outKeys[newpost]= key;  // killing line !!!
    out_key[newpost] = org_value;

#ifdef PERMUT
    indices_out[newpost] = indices_in[k];
#endif

    newpos++;
    loc_histo[shortkey * items + it] = newpos;
  }
}

// perform a parallel prefix sum (a scan) on the local histograms
// (see Blelloch 1990) each workitem worries about two memories
// see also http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
__kernel void radixsort_scanhistograms(__global int *histo, __local int *temp,
                                       __global int *globsum)
{
  int it = get_local_id(0);
  int ig = get_global_id(0);
  int decale = 1;
  int n = get_local_size(0) * 2;
  int gr = get_group_id(0);

  // load input into local memory
  // up sweep phase
  temp[2 * it] = histo[2 * ig];
  temp[2 * it + 1] = histo[2 * ig + 1];

  // parallel prefix sum (algorithm of Blelloch 1990)
  for (int d = n >> 1; d > 0; d >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (it < d)
    {
      int ai = decale * (2 * it + 1) - 1;
      int bi = decale * (2 * it + 2) - 1;
      temp[bi] += temp[ai];
    }
    decale *= 2;
  }

  // store the last element in the global sum vector
  // (maybe used in the next step for constructing the global scan)
  // clear the last element
  if (it == 0)
  {
    globsum[gr] = temp[n - 1];
    temp[n - 1] = 0;
  }

  // down sweep phase
  for (int d = 1; d < n; d *= 2)
  {
    decale >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (it < d)
    {
      int ai = decale * (2 * it + 1) - 1;
      int bi = decale * (2 * it + 2) - 1;

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // write results to device memory

  histo[2 * ig] = temp[2 * it];
  histo[2 * ig + 1] = temp[2 * it + 1];

  barrier(CLK_GLOBAL_MEM_FENCE);
}

// use the global sum for updating the local histograms
// each work item updates two values
__kernel void radixsort_pastehistograms(__global int *histo, __global int *globsum)
{
  int ig = get_global_id(0);
  int gr = get_group_id(0);

  int s;

  s = globsum[gr];

  // write results to device memory
  histo[2 * ig] += s;
  histo[2 * ig + 1] += s;

  barrier(CLK_GLOBAL_MEM_FENCE);
}
