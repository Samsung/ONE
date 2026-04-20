# problem 1

Firmware Version

- Project: Samsung ONE

- Repository: https://github.com/Samsung/ONE

- Verified commit (full): 093fca849b1505705cf72c1a5f99c8bc2cb9eabd

- Verified commit (short): 093fca84

- Verified commit date: 2026-04-15 12:29:04 +0900

- Validation date: 2026-04-17

- Toolchain used for reproduction:

  - circle-verify (BuildID: 1fc0053950cb395bdea5a1141d86dba2ae318eeb)

  - circle-inspect (BuildID: 07d94c72d04d07a634766bef4a13df6a715a5aff)

  - circle2circle v1.31.0 (BuildID: d03c758d20e62ec111e56743ebab8e60c2cac963)

- Note for Samsung Mobile mapping: this report was validated on the OSS ONE commit above.

  Exact Galaxy model / One UI firmware mapping should be checked against Samsung’s internal integration branch.



Description

I reproduced this issue on https://github.com/Samsung/ONE with a model that contains invalid

STRING tensor offsets.



The model passes `circle-verify`, but crashes in `circle2circle` when string constants are read.

So the malformed offset table is not blocked early enough.



Code path

In `compiler/luci/import/src/Nodes/CircleConst.cpp`:

- `int32_t start = offsets[i];`

- `int32_t next = offsets[i + 1];`

- `std::string value(data + start, next - start);`



If offsets are negative or inconsistent, pointer/length construction becomes invalid.



Reproduction

1. Generate and run with `poc/reproduce.sh`

2. Or run directly:

   - `circle-verify poc_string_offset_int32min.circle`

   - `circle-inspect --constants --tensor_dtype --tensor_shape poc_string_offset_int32min.circle`

   - `circle2circle --remove_redundant_reshape poc_string_offset_int32min.circle out.circle`



Observed result

- `circle-verify`: PASS (exit 0)

- `circle-inspect`: PASS (exit 0)

- `circle2circle`: SIGSEGV (exit 139)

- Runtime evidence: `evidence/recheck_runtime_int32min_crash.log`

- ASAN evidence: `evidence/recheck_poc_string_offset_asan.log`



Security impact

A malicious model can bypass verifier checks and trigger process termination in import/optimization

stage, resulting in reliable denial of service.



Suggested fix

- Validate STRING offsets as non-negative, monotonic, and bounded within data buffer.

- Reject malformed offset tables before pointer arithmetic.

- Handle invalid input with explicit error returns instead of process crash.



Attached files

- `poc/poc_string_offset_int32min.circle`

- `poc/poc_string_offset_int32min.circle.json`

- `poc/reproduce.sh`

- `evidence/recheck_runtime_int32min_crash.log`

- `evidence/recheck_poc_string_offset_asan.log`

- `evidence/recheck_code_path.txt`

- `evidence/version_info.txt`


# problem 2

Firmware Version

- Project: Samsung ONE

- Repository: https://github.com/Samsung/ONE

- Verified commit (full): 093fca849b1505705cf72c1a5f99c8bc2cb9eabd

- Verified commit (short): 093fca84

- Verified commit date: 2026-04-15 12:29:04 +0900

- Validation date: 2026-04-17

- Toolchain used for reproduction:

  - circle-verify (BuildID: 1fc0053950cb395bdea5a1141d86dba2ae318eeb)

  - circle-inspect (BuildID: 07d94c72d04d07a634766bef4a13df6a715a5aff)

  - circle2circle v1.31.0 (BuildID: d03c758d20e62ec111e56743ebab8e60c2cac963)

- Note for Samsung Mobile mapping: this report was validated on the OSS ONE commit above.

  Exact Galaxy model / One UI firmware mapping should be checked against Samsung’s internal integration branch.



Description

I verified this issue on https://github.com/Samsung/ONE using the commit listed above.

A crafted model with an out-of-range `opcode_index` is accepted by `circle-verify`, but crashes

in later processing (`circle-inspect`, `circle2circle`).



In short, malformed operator metadata crosses the verifier boundary and fails only when

operator code lookup happens downstream.



Code locations

- `runtime/onert/core/src/loader/BaseLoader.h:116`

  - `_domain_model->operator_codes()->Get(op->opcode_index())`

- `compiler/mio-circle/src/Reader.cpp:164`

  - `assert(index < _op_codes.size())`

- `compiler/luci/import/src/CircleReader.cpp:355`

  - `assert(index < op_codes.size())`



Reproduction

1. Prepare PoC model:

   - `poc/poc_opcode_index_oob.circle`

2. Run:

   - `circle-verify poc_opcode_index_oob.circle`

   - `circle-inspect --operators poc_opcode_index_oob.circle`

   - `circle2circle --remove_redundant_reshape poc_opcode_index_oob.circle out.circle`



Observed result

- `circle-verify`: PASS (exit 0)

- `circle-inspect --operators`: abort (exit 134)

- `circle2circle --remove_redundant_reshape`: abort (exit 134)

- Evidence log: `evidence/recheck_runtime_opcode_index_oob.log`



Security impact

This is a reliable denial-of-service condition in model-processing pipelines that trust verifier pass

as a safety gate. A malicious model can survive initial validation and terminate follow-up stages.



Suggested fix

- Enforce explicit bounds validation for `opcode_index` before every operator-code lookup.

- Apply the same check consistently across verifier/import/runtime paths.

- Return controlled parser/validation errors instead of aborting.



Attached files

- `poc/poc_opcode_index_oob.circle`

- `poc/poc_opcode_index_oob.circle.json`

- `poc/reproduce.sh`

- `evidence/recheck_runtime_opcode_index_oob.log`

- `evidence/recheck_code_path.txt`

- `evidence/version_info.txt`

# problem 3

## Summary



Multiple integer overflow vulnerabilities (CWE-190) were identified in Samsung ONE's luci-interpreter component that can lead to heap buffer overflow (CWE-122) when processing malicious Circle model files.



## Root Cause



The Shape::num_elements() function returns int32_t and can overflow when tensor dimensions are large:



File: compiler/luci-interpreter/include/luci_interpreter/core/Tensor.h (lines 53-61)

```cpp

int32_t num_elements() const

{

    int32_t result = 1;

    for (const int32_t dim : _dims)

    {

        result *= dim;  // Integer overflow possible!

    }

    return result;

}

```



Samsung is aware of this issue - there's a comment "// TODO Replace num_elements" and a safe version large_num_elements() exists, but multiple locations still use the vulnerable num_elements().



## Vulnerable Locations (7 instances)



### 1. BuddyMemoryManager.cpp (Lines 46-47)

```cpp

const int32_t num_elements = tensor.shape().num_elements();

auto size = num_elements * element_size;

```



### 2. ExpandDims.cpp (Lines 83-84)

```cpp

const int32_t num_elements = input()->shape().num_elements();

std::memcpy(output_data, input_data, num_elements * element_size);

```



### 3. Reshape.cpp (Lines 104-105)

```cpp

const int32_t num_elements = input()->shape().num_elements();

std::memcpy(output_data, input_data, num_elements * element_size);

```



### 4. While.cpp (Lines 37-39)

```cpp

const int32_t num_elements = src[i]->shape().num_elements();

std::memcpy(dst[i]->data<void>(), src[i]->data<void>(), num_elements * element_size);

```



### 5. If.cpp (Lines 73, 89)

```cpp

std::memcpy(graph_inputs[i]->data<void>(), input(i)->data<void>(), num_elements * element_size);

```



### 6. TransposeConv.cpp (Lines 205, 299)

```cpp

std::memset(scratch_data, 0, scratch_tensor->shape().num_elements() * sizeof(int32_t));

```



### 7. GraphLoader.cpp (Line 44)

```cpp

*data_size = num_elements * element_size;

```



## Attack Scenario



1. Attacker crafts malicious Circle model with tensor dimensions causing overflow

   - Example: dims = [65536, 65536] = 4,294,967,296 overflows to 0 in int32_t

   - Example: dims = [65536, 32768] = 2,147,483,648 overflows to negative

2. Victim loads model using Samsung ONE runtime (e.g., on-device AI inference)

3. Memory allocated with truncated/incorrect size

4. memcpy/memset writes beyond allocated buffer

5. Heap corruption  potential code execution



## Steps to Reproduce



1. Clone Samsung ONE repository: https://github.com/Samsung/ONE

2. Review file: compiler/luci-interpreter/include/luci_interpreter/core/Tensor.h

3. Observe num_elements() returns int32_t (line 53)

4. Observe large_num_elements() returns int64_t (line 64) with comment "// TODO Replace num_elements"

5. Search for usages: grep -rn "\.num_elements()" compiler/luci-interpreter/src/

6. Confirm multiple locations use vulnerable num_elements() instead of large_num_elements()



## Impact



- Heap Buffer Overflow via memcpy/memset with truncated size

- Potential Remote Code Execution when processing untrusted Circle model files

- Affects Samsung devices using ONE for on-device AI inference



## CVSS Score



8.1 (High) - AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H



## Recommended Fix



Replace all instances of num_elements() with large_num_elements() and add overflow checks:

```cpp

// Before (vulnerable):

const int32_t num_elements = input()->shape().num_elements();

std::memcpy(output_data, input_data, num_elements * element_size);



// After (fixed):

const int64_t num_elements = input()->shape().large_num_elements();

const int64_t total_size = num_elements * static_cast<int64_t>(element_size);

if (total_size > SIZE_MAX || num_elements < 0) {

    throw std::runtime_error("Integer overflow in size calculation");

}

std::memcpy(output_data, input_data, static_cast<size_t>(total_size));

```



## Affected Files



1. compiler/luci-interpreter/src/BuddyMemoryManager.cpp

2. compiler/luci-interpreter/src/kernels/ExpandDims.cpp

3. compiler/luci-interpreter/src/kernels/Reshape.cpp

4. compiler/luci-interpreter/src/kernels/While.cpp

5. compiler/luci-interpreter/src/kernels/If.cpp

6. compiler/luci-interpreter/src/kernels/TransposeConv.cpp

7. compiler/luci-interpreter/src/loader/GraphLoader.cpp





## References



- CWE-190: Integer Overflow or Wraparound

- CWE-122: Heap-based Buffer Overflow

- Similar: CVE-2022-29216 (TensorFlow integer overflow)