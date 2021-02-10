#ifndef HALIDE_INTERNAL_ADD_IMAGE_CHECKS_H
#define HALIDE_INTERNAL_ADD_IMAGE_CHECKS_H

/** \file
 *
 * Defines the lowering pass that adds the assertions that validate
 * input and output buffers.
 */

#ifndef HALIDE_IR_H
#define HALIDE_IR_H

/** \file
 * Subtypes for Halide expressions (\ref Halide::Expr) and statements (\ref Halide::Internal::Stmt)
 */

#include <string>
#include <vector>

#ifndef HALIDE_DEBUG_H
#define HALIDE_DEBUG_H

/** \file
 * Defines functions for debug logging during code generation.
 */

#include <iostream>
#include <string>
#include <stdlib.h>

#ifndef HALIDE_INTROSPECTION_H
#define HALIDE_INTROSPECTION_H

#include <string>
#include <iostream>
#include <stdint.h>

// Always use assert, even if llvm-config defines NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#include <assert.h>
#define NDEBUG
#else
#include <assert.h>
#endif

#ifndef HALIDE_UTIL_H
#define HALIDE_UTIL_H

/** \file
 * Various utility functions used internally Halide. */

#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <cstring>

#ifndef EXPORT
#if defined(_MSC_VER)
#ifdef Halide_EXPORTS
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif
#else
#define EXPORT __attribute__((visibility("default")))
#endif
#endif

// If we're in user code, we don't want certain functions to be inlined.
#if defined(COMPILING_HALIDE) || defined(BUILDING_PYTHON)
#define NO_INLINE
#else
#ifdef _WIN32
#define NO_INLINE __declspec(noinline)
#else
#define NO_INLINE __attribute__((noinline))
#endif
#endif

// On windows, Halide needs a larger stack than the default MSVC provides
#ifdef _MSC_VER
#pragma comment(linker, "/STACK:8388608,1048576")
#endif

namespace Halide {
namespace Internal {

/** An aggressive form of reinterpret cast used for correct type-punning. */
template<typename DstType, typename SrcType>
DstType reinterpret_bits(const SrcType &src) {
    static_assert(sizeof(SrcType) == sizeof(DstType), "Types must be same size");
    DstType dst;
    memcpy(&dst, &src, sizeof(SrcType));
    return dst;
}

/** Make a unique name for an object based on the name of the stack
 * variable passed in. If introspection isn't working or there are no
 * debug symbols, just uses unique_name with the given prefix. */
EXPORT std::string make_entity_name(void *stack_ptr, const std::string &type, char prefix);

/** Get value of an environment variable. Returns its value
 * is defined in the environment. If the var is not defined, an empty string
 * is returned.
 */
EXPORT std::string get_env_variable(char const *env_var_name);

/** Get the name of the currently running executable. Platform-specific.
 * If program name cannot be retrieved, function returns an empty string. */
EXPORT std::string running_program_name();

/** Generate a unique name starting with the given prefix. It's unique
 * relative to all other strings returned by unique_name in this
 * process.
 *
 * The single-character version always appends a numeric suffix to the
 * character.
 *
 * The string version will either return the input as-is (with high
 * probability on the first time it is called with that input), or
 * replace any existing '$' characters with underscores, then add a
 * '$' sign and a numeric suffix to it.
 *
 * Note that unique_name('f') therefore differs from
 * unique_name("f"). The former returns something like f123, and the
 * latter returns either f or f$123.
 */
// @{
EXPORT std::string unique_name(char prefix);
EXPORT std::string unique_name(const std::string &prefix);
// @}

/** Test if the first string starts with the second string */
EXPORT bool starts_with(const std::string &str, const std::string &prefix);

/** Test if the first string ends with the second string */
EXPORT bool ends_with(const std::string &str, const std::string &suffix);

/** Replace all matches of the second string in the first string with the last string */
EXPORT std::string replace_all(const std::string &str, const std::string &find, const std::string &replace);

/** Split the source string using 'delim' as the divider. */
EXPORT std::vector<std::string> split_string(const std::string &source, const std::string &delim);

/** Perform a left fold of a vector. Returns a default-constructed
 * vector element if the vector is empty. Similar to std::accumulate
 * but with a less clunky syntax. */
template<typename T, typename Fn>
T fold_left(const std::vector<T> &vec, Fn f) {
    T result;
    if (vec.empty()) {
        return result;
    }
    result = vec[0];
    for (size_t i = 1; i < vec.size(); i++) {
        result = f(result, vec[i]);
    }
    return result;
}

/** Returns a right fold of a vector. Returns a default-constructed
 * vector element if the vector is empty. */
template<typename T, typename Fn>
T fold_right(const std::vector<T> &vec, Fn f) {
    T result;
    if (vec.empty()) {
        return result;
    }
    result = vec.back();
    for (size_t i = vec.size()-1; i > 0; i--) {
        result = f(vec[i-1], result);
    }
    return result;
}

template <typename T1, typename T2, typename T3, typename T4 >
inline NO_INLINE void collect_paired_args(std::vector<std::pair<T1, T2>> &collected_args,
                                     const T3 &a1, const T4 &a2) {
    collected_args.push_back(std::pair<T1, T2>(a1, a2));
}

template <typename T1, typename T2, typename T3, typename T4, typename ...Args>
inline NO_INLINE void collect_paired_args(std::vector<std::pair<T1, T2>> &collected_args,
                                   const T3 &a1, const T4 &a2, Args&&... args) {
    collected_args.push_back(std::pair<T1, T2>(a1, a2));
    collect_paired_args(collected_args, std::forward<Args>(args)...);
}

template<typename... T>
struct meta_and : std::true_type {};

template<typename T1, typename... Args>
struct meta_and<T1, Args...> : std::integral_constant<bool, T1::value && meta_and<Args...>::value> {};

template<typename... T>
struct meta_or : std::false_type {};

template<typename T1, typename... Args>
struct meta_or<T1, Args...> : std::integral_constant<bool, T1::value || meta_or<Args...>::value> {};

template<typename To, typename... Args>
struct all_are_convertible : meta_and<std::is_convertible<Args, To>...> {};

/** Returns base name and fills in namespaces, outermost one first in vector. */
EXPORT std::string extract_namespaces(const std::string &name, std::vector<std::string> &namespaces);

struct FileStat {
    uint64_t file_size;
    uint32_t mod_time;  // Unix epoch time
    uint32_t uid;
    uint32_t gid;
    uint32_t mode;
};

/** Create a unique file with a name of the form prefixXXXXXsuffix in an arbitrary
 * (but writable) directory; this is typically /tmp, but the specific
 * location is not guaranteed. (Note that the exact form of the file name
 * may vary; in particular, the suffix may be ignored on Windows.)
 * The file is created (but not opened), thus this can be called from
 * different threads (or processes, e.g. when building with parallel make)
 * without risking collision. Note that if this file is used as a temporary
 * file, the caller is responsibly for deleting it. Neither the prefix nor suffix
 * may contain a directory separator.
 */
EXPORT std::string file_make_temp(const std::string &prefix, const std::string &suffix);

/** Create a unique directory in an arbitrary (but writable) directory; this is
 * typically somewhere inside /tmp, but the specific location is not guaranteed.
 * The directory will be empty (i.e., this will never return /tmp itself,
 * but rather a new directory inside /tmp). The caller is responsible for removing the
 * directory after use.
 */
EXPORT std::string dir_make_temp();

/** Wrapper for access(). Quietly ignores errors. */
EXPORT bool file_exists(const std::string &name);

/** assert-fail if the file doesn't exist. useful primarily for testing purposes. */
EXPORT void assert_file_exists(const std::string &name);

/** assert-fail if the file DOES exist. useful primarily for testing purposes. */
EXPORT void assert_no_file_exists(const std::string &name);

/** Wrapper for unlink(). Asserts upon error. */
EXPORT void file_unlink(const std::string &name);

/** Wrapper for unlink(). Quietly ignores errors. */
EXPORT void file_unlink(const std::string &name);

/** Ensure that no file with this path exists. If such a file
 * exists and cannot be removed, assert-fail. */
EXPORT void ensure_no_file_exists(const std::string &name);

/** Wrapper for rmdir(). Asserts upon error. */
EXPORT void dir_rmdir(const std::string &name);

/** Wrapper for stat(). Asserts upon error. */
EXPORT FileStat file_stat(const std::string &name);

/** A simple utility class that creates a temporary file in its ctor and
 * deletes that file in its dtor; this is useful for temporary files that you
 * want to ensure are deleted when exiting a certain scope. Since this is essentially
 * just an RAII wrapper around file_make_temp() and file_unlink(), it has the same
 * failure modes (i.e.: assertion upon error).
 */
class TemporaryFile final {
public:
    TemporaryFile(const std::string &prefix, const std::string &suffix)
        : temp_path(file_make_temp(prefix, suffix)), do_unlink(true) {}
    const std::string &pathname() const { return temp_path; }
    ~TemporaryFile() { if (do_unlink) { file_unlink(temp_path); } }
    // You can call this if you want to defeat the automatic deletion;
    // this is rarely what you want to do (since it defeats the purpose
    // of this class), but can be quite handy for debugging purposes.
    void detach() { do_unlink = false; }
private:
    const std::string temp_path;
    bool do_unlink;
    TemporaryFile(const TemporaryFile &) = delete;
    void operator=(const TemporaryFile &) = delete;
};

/** Routines to test if math would overflow for signed integers with
 * the given number of bits. */
// @{
bool add_would_overflow(int bits, int64_t a, int64_t b);
bool sub_would_overflow(int bits, int64_t a, int64_t b);
bool mul_would_overflow(int bits, int64_t a, int64_t b);
// @}

// Wrappers for some C++14-isms that are useful and trivially implementable
// in C++11; these are defined in the Halide::Internal namespace. If we
// are compiling under C++14 or later, we just use the standard implementations
// rather than our own.
#if __cplusplus >= 201402L

// C++14: Use the standard implementations
using std::integer_sequence;
using std::make_integer_sequence;
using std::index_sequence;
using std::make_index_sequence;

#else

// C++11: std::integer_sequence (etc) is standard in C++14 but not C++11, but
// is easily written in C++11. This is a simple version that could
// probably be improved.

template<typename T, T... Ints>
struct integer_sequence {
    static constexpr size_t size() { return sizeof...(Ints); }
};

template<typename T>
struct next_integer_sequence;

template<typename T, T... Ints>
struct next_integer_sequence<integer_sequence<T, Ints...>> {
    using type = integer_sequence<T, Ints..., sizeof...(Ints)>;
};

template<typename T, T I, T N>
struct make_integer_sequence_helper {
    using type = typename next_integer_sequence<
        typename make_integer_sequence_helper<T, I+1, N>::type
    >::type;
};

template<typename T, T N>
struct make_integer_sequence_helper<T, N, N> {
    using type = integer_sequence<T>;
};

template<typename T, T N>
using make_integer_sequence = typename make_integer_sequence_helper<T, 0, N>::type;

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template<size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

#endif

}  // namespace Internal
}  // namespace Halide

#endif

/** \file
 *
 * Defines methods for introspecting in C++. Relies on DWARF debugging
 * metadata, so the compilation unit that uses this must be compiled
 * with -g.
 */

namespace Halide {
namespace Internal {

namespace Introspection {
/** Get the name of a stack variable from its address. The stack
 * variable must be in a compilation unit compiled with -g to
 * work. The expected type helps distinguish between variables at the
 * same address, e.g a class instance vs its first member. */
EXPORT std::string get_variable_name(const void *, const std::string &expected_type);

/** Register an untyped heap object. Derive type information from an
 * introspectable pointer to a pointer to a global object of the same
 * type. Not thread-safe. */
EXPORT void register_heap_object(const void *obj, size_t size, const void *helper);

/** Deregister a heap object. Not thread-safe. */
EXPORT void deregister_heap_object(const void *obj, size_t size);

/** Return the address of a global with type T *. Call this to
 * generate something to pass as the last argument to
 * register_heap_object.
 */
template<typename T>
const void *get_introspection_helper() {
    static T *introspection_helper = nullptr;
    return &introspection_helper;
}

/** Get the source location in the call stack, skipping over calls in
 * the Halide namespace. */
EXPORT std::string get_source_location();

// This gets called automatically by anyone who includes Halide.h by
// the code below. It tests if this functionality works for the given
// compilation unit, and disables it if not.
EXPORT void test_compilation_unit(bool (*test)(bool (*)(const void *, const std::string &)),
                                  bool (*test_a)(const void *, const std::string &),
                                  void (*calib)());
}

}
}


// This code verifies that introspection is working before relying on
// it. The definitions must appear in Halide.h, but they should not
// appear in libHalide itself. They're defined as static so that clients
// can include Halide.h multiple times without link errors.
#ifndef COMPILING_HALIDE

namespace Halide {
namespace Internal {
static bool check_introspection(const void *var, const std::string &type,
                                const std::string &correct_name,
                                const std::string &correct_file, int line) {
    std::string correct_loc = correct_file + ":" + std::to_string(line);
    std::string loc = Introspection::get_source_location();
    std::string name = Introspection::get_variable_name(var, type);
    return name == correct_name && loc == correct_loc;
}
}
}

namespace HalideIntrospectionCanary {

// A function that acts as a signpost. By taking it's address and
// comparing it to the program counter listed in the debugging info,
// we can calibrate for any offset between the debugging info and the
// actual memory layout where the code was loaded.
static void offset_marker() {
    std::cerr << "You should not have called this function\n";
}

struct A {
    int an_int;

    class B {
        int private_member;
    public:
        float a_float;
        A *parent;
        B() : private_member(17) {
            a_float = private_member * 2.0f;
        }
    };

    B a_b;

    A() {
        a_b.parent = this;
    }

    bool test(const std::string &my_name);
};

static bool test_a(const void *a_ptr, const std::string &my_name) {
    const A *a = (const A *)a_ptr;
    bool success = true;
    success &= Halide::Internal::check_introspection(&a->an_int, "int", my_name + ".an_int", __FILE__ , __LINE__);
    success &= Halide::Internal::check_introspection(&a->a_b, "HalideIntrospectionCanary::A::B", my_name + ".a_b", __FILE__ , __LINE__);
    success &= Halide::Internal::check_introspection(&a->a_b.parent, "HalideIntrospectionCanary::A *", my_name + ".a_b.parent", __FILE__ , __LINE__);
    success &= Halide::Internal::check_introspection(&a->a_b.a_float, "float", my_name + ".a_b.a_float", __FILE__ , __LINE__);
    success &= Halide::Internal::check_introspection(a->a_b.parent, "HalideIntrospectionCanary::A", my_name, __FILE__ , __LINE__);
    return success;
}

static bool test(bool (*f)(const void *, const std::string &)) {
    A a1, a2;

    // Call via pointer to prevent inlining.
    return f(&a1, "a1") && f(&a2, "a2");
}

// Run the tests, and calibrate for the PC offset at static initialization time.
namespace {
struct TestCompilationUnit {
    TestCompilationUnit() {
        Halide::Internal::Introspection::test_compilation_unit(&test, &test_a, &offset_marker);
    }
};
}

static TestCompilationUnit test_object;

}

#endif

#endif

namespace Halide {

struct Expr;
struct Type;
// Forward declare some things from IRPrinter, which we can't include yet.
EXPORT std::ostream &operator<<(std::ostream &stream, const Expr &);
EXPORT std::ostream &operator<<(std::ostream &stream, const Type &);

class Module;
EXPORT std::ostream &operator<<(std::ostream &stream, const Module &);

namespace Internal {

struct Stmt;
EXPORT std::ostream &operator<<(std::ostream &stream, const Stmt &);

struct LoweredFunc;
EXPORT std::ostream &operator << (std::ostream &, const LoweredFunc &);

/** For optional debugging during codegen, use the debug class as
 * follows:
 *
 \code
 debug(verbosity) << "The expression is " << expr << std::endl;
 \endcode
 *
 * verbosity of 0 always prints, 1 should print after every major
 * stage, 2 should be used for more detail, and 3 should be used for
 * tracing everything that occurs. The verbosity with which to print
 * is determined by the value of the environment variable
 * HL_DEBUG_CODEGEN
 */

class debug {
    const bool logging;

public:
    debug(int verbosity) : logging(verbosity <= debug_level()) {}

    template<typename T>
    debug &operator<<(T&& x) {
        if (logging) {
            std::cerr << std::forward<T>(x);
        }
        return *this;
    }

    EXPORT static int debug_level();
};

}
}

#endif
#ifndef HALIDE_ERROR_H
#define HALIDE_ERROR_H

#include <sstream>
#include <stdexcept>

#ifndef HALIDE_HALIDERUNTIME_H
#define HALIDE_HALIDERUNTIME_H

#ifndef COMPILING_HALIDE_RUNTIME
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#else
#error "COMPILING_HALIDE_RUNTIME should never be defined for Halide.h"
#endif

#ifdef __cplusplus
// Forward declare type to allow naming typed handles.
// See Type.h for documentation.
template<typename T> struct halide_handle_traits;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Note that you should not use "inline" along with HALIDE_ALWAYS_INLINE;
// it is not necessary, and may produce warnings for some build configurations.
#ifdef _MSC_VER
#define HALIDE_ALWAYS_INLINE __forceinline
#else
#define HALIDE_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

/** \file
 *
 * This file declares the routines used by Halide internally in its
 * runtime. On platforms that support weak linking, these can be
 * replaced with user-defined versions by defining an extern "C"
 * function with the same name and signature.
 *
 * When doing Just In Time (JIT) compilation methods on the Func being
 * compiled must be called instead. The corresponding methods are
 * documented below.
 *
 * All of these functions take a "void *user_context" parameter as their
 * first argument; if the Halide kernel that calls back to any of these
 * functions has been compiled with the UserContext feature set on its Target,
 * then the value of that pointer passed from the code that calls the
 * Halide kernel is piped through to the function.
 *
 * Some of these are also useful to call when using the default
 * implementation. E.g. halide_shutdown_thread_pool.
 *
 * Note that even on platforms with weak linking, some linker setups
 * may not respect the override you provide. E.g. if the override is
 * in a shared library and the halide object files are linked directly
 * into the output, the builtin versions of the runtime functions will
 * be called. See your linker documentation for more details. On
 * Linux, LD_DYNAMIC_WEAK=1 may help.
 *
 */

// Forward-declare to suppress warnings if compiling as C.
struct halide_buffer_t;
struct buffer_t;

/** Print a message to stderr. Main use is to support tracing
 * functionality, print, and print_when calls. Also called by the default
 * halide_error.  This function can be replaced in JITed code by using
 * halide_custom_print and providing an implementation of halide_print
 * in AOT code. See Func::set_custom_print.
 */
// @{
extern void halide_print(void *user_context, const char *);
extern void halide_default_print(void *user_context, const char *);
typedef void (*halide_print_t)(void *, const char *);
extern halide_print_t halide_set_custom_print(halide_print_t print);
// @}

/** Halide calls this function on runtime errors (for example bounds
 * checking failures). This function can be replaced in JITed code by
 * using Func::set_error_handler, or in AOT code by calling
 * halide_set_error_handler. In AOT code on platforms that support
 * weak linking (i.e. not Windows), you can also override it by simply
 * defining your own halide_error.
 */
// @{
extern void halide_error(void *user_context, const char *);
extern void halide_default_error(void *user_context, const char *);
typedef void (*halide_error_handler_t)(void *, const char *);
extern halide_error_handler_t halide_set_error_handler(halide_error_handler_t handler);
// @}

/** Cross-platform mutex. These are allocated statically inside the
 * runtime, hence the fixed size. They must be initialized with
 * zero. The first time halide_mutex_lock is called, the lock must be
 * initialized in a thread safe manner. This incurs a small overhead
 * for a once mechanism, but makes the lock reliably easy to setup and
 * use without depending on e.g. C++ constructor logic.
 */
struct halide_mutex {
    uint64_t _private[8];
};

/** A basic set of mutex and condition variable functions, which call
 * platform specific code for mutual exclusion. Equivalent to posix
 * calls. Mutexes should initially be set to zero'd memory. Any
 * resources required are created on first lock. Calling destroy
 * re-zeros the memory.
 */
//@{
extern void halide_mutex_lock(struct halide_mutex *mutex);
extern void halide_mutex_unlock(struct halide_mutex *mutex);
extern void halide_mutex_destroy(struct halide_mutex *mutex);

//@}

/** Define halide_do_par_for to replace the default thread pool
 * implementation. halide_shutdown_thread_pool can also be called to
 * release resources used by the default thread pool on platforms
 * where it makes sense. (E.g. On Mac OS, Grand Central Dispatch is
 * used so %Halide does not own the threads backing the pool and they
 * cannot be released.)  See Func::set_custom_do_task and
 * Func::set_custom_do_par_for. Should return zero if all the jobs
 * return zero, or an arbitrarily chosen return value from one of the
 * jobs otherwise.
 */
//@{
typedef int (*halide_task_t)(void *user_context, int task_number, uint8_t *closure);
extern int halide_do_par_for(void *user_context,
                             halide_task_t task,
                             int min, int size, uint8_t *closure);
extern void halide_shutdown_thread_pool();
typedef int (*halide_task_64_t)(void *user_context, int64_t task_number, uint8_t *closure);
extern int halide_do_par_for_64(void *user_context,
                             halide_task_64_t task,
                             int64_t min, int64_t size, uint8_t *closure);
extern void halide_shutdown_thread_pool_64();
//@}

/** Set a custom method for performing a parallel for loop. Returns
 * the old do_par_for handler. */
typedef int (*halide_do_par_for_t)(void *, halide_task_t, int, int, uint8_t*);
extern halide_do_par_for_t halide_set_custom_do_par_for(halide_do_par_for_t do_par_for);
typedef int (*halide_do_par_for_64_t)(void *, halide_task_64_t, int64_t, int64_t, uint8_t*);
extern halide_do_par_for_64_t halide_set_custom_do_par_for_64(halide_do_par_for_64_t do_par_for);

/** If you use the default do_par_for, you can still set a custom
 * handler to perform each individual task. Returns the old handler. */
//@{
typedef int (*halide_do_task_t)(void *, halide_task_t, int, uint8_t *);
extern halide_do_task_t halide_set_custom_do_task(halide_do_task_t do_task);
extern int halide_do_task(void *user_context, halide_task_t f, int idx,
                          uint8_t *closure);
typedef int (*halide_do_task_64_t)(void *, halide_task_64_t, int64_t, uint8_t *);
extern halide_do_task_64_t halide_set_custom_do_task_64(halide_do_task_64_t do_task);
extern int halide_do_task_64(void *user_context, halide_task_64_t f, int64_t idx,
                             uint8_t *closure);
//@}

/** The default versions of do_task and do_par_for. Can be convenient
 * to call from overrides in certain circumstances. */
// @{
extern int halide_default_do_par_for(void *user_context,
                                     halide_task_t task,
                                     int min, int size, uint8_t *closure);
extern int halide_default_do_task(void *user_context, halide_task_t f, int idx,
                                  uint8_t *closure);
extern int halide_default_do_par_for_64(void *user_context, halide_task_64_t task,
                                     int64_t min, int64_t size, uint8_t *closure);
extern int halide_default_do_task_64(void *user_context, halide_task_64_t f, int64_t idx,
                                  uint8_t *closure);
// @}

struct halide_thread;

/** Spawn a thread. Returns a handle to the thread for the purposes of
 * joining it. The thread must be joined in order to clean up any
 * resources associated with it. */
extern struct halide_thread *halide_spawn_thread(void (*f)(void *), void *closure);

/** Join a thread. */
extern void halide_join_thread(struct halide_thread *);

/** Set the number of threads used by Halide's thread pool. Returns
 * the old number.
 *
 * n < 0  : error condition
 * n == 0 : use a reasonable system default (typically, number of cpus online).
 * n == 1 : use exactly one thread; this will always enforce serial execution
 * n > 1  : use a pool of exactly n threads.
 *
 * Note that the default iOS and OSX behavior will treat n > 1 like n == 0;
 * that is, any positive value other than 1 will use a system-determined number
 * of threads.
 *
 * (Note that this is only guaranteed when using the default implementations
 * of halide_do_par_for(); custom implementations may completely ignore values
 * passed to halide_set_num_threads().)
 */
extern int halide_set_num_threads(int n);
extern int halide_set_num_threads_64(int n);

/** Halide calls these functions to allocate and free memory. To
 * replace in AOT code, use the halide_set_custom_malloc and
 * halide_set_custom_free, or (on platforms that support weak
 * linking), simply define these functions yourself. In JIT-compiled
 * code use Func::set_custom_allocator.
 *
 * If you override them, and find yourself wanting to call the default
 * implementation from within your override, use
 * halide_default_malloc/free.
 *
 * Note that halide_malloc must return a pointer aligned to the
 * maximum meaningful alignment for the platform for the purpose of
 * vector loads and stores. The default implementation uses 32-byte
 * alignment, which is safe for arm and x86. Additionally, it must be
 * safe to read at least 8 bytes before the start and beyond the
 * end.
 */
//@{
extern void *halide_malloc(void *user_context, size_t x);
extern void halide_free(void *user_context, void *ptr);
extern void *halide_default_malloc(void *user_context, size_t x);
extern void halide_default_free(void *user_context, void *ptr);
typedef void *(*halide_malloc_t)(void *, size_t);
typedef void (*halide_free_t)(void *, void *);
extern halide_malloc_t halide_set_custom_malloc(halide_malloc_t user_malloc);
extern halide_free_t halide_set_custom_free(halide_free_t user_free);
//@}

/** Halide calls these functions to interact with the underlying
 * system runtime functions. To replace in AOT code on platforms that
 * support weak linking, define these functions yourself, or use
 * the halide_set_custom_load_library() and halide_set_custom_get_library_symbol()
 * functions. In JIT-compiled code, use JITSharedRuntime::set_default_handlers().
 *
 * halide_load_library and halide_get_library_symbol are equivalent to
 * dlopen and dlsym. halide_get_symbol(sym) is equivalent to
 * dlsym(RTLD_DEFAULT, sym).
 */
//@{
extern void *halide_get_symbol(const char *name);
extern void *halide_load_library(const char *name);
extern void *halide_get_library_symbol(void *lib, const char *name);
extern void *halide_default_get_symbol(const char *name);
extern void *halide_default_load_library(const char *name);
extern void *halide_default_get_library_symbol(void *lib, const char *name);
typedef void *(*halide_get_symbol_t)(const char *name);
typedef void *(*halide_load_library_t)(const char *name);
typedef void *(*halide_get_library_symbol_t)(void *lib, const char *name);
extern halide_get_symbol_t halide_set_custom_get_symbol(halide_get_symbol_t user_get_symbol);
extern halide_load_library_t halide_set_custom_load_library(halide_load_library_t user_load_library);
extern halide_get_library_symbol_t halide_set_custom_get_library_symbol(halide_get_library_symbol_t user_get_library_symbol);
//@}

/** Called when debug_to_file is used inside %Halide code.  See
 * Func::debug_to_file for how this is called
 *
 * Cannot be replaced in JITted code at present.
 */
extern int32_t halide_debug_to_file(void *user_context, const char *filename,
                                    int32_t type_code,
                                    struct halide_buffer_t *buf);

/** Types in the halide type system. They can be ints, unsigned ints,
 * or floats (of various bit-widths), or a handle (which is always 64-bits).
 * Note that the int/uint/float values do not imply a specific bit width
 * (the bit width is expected to be encoded in a separate value).
 */
typedef enum halide_type_code_t
#if __cplusplus >= 201103L
: uint8_t
#endif
{
    halide_type_int = 0,   //!< signed integers
    halide_type_uint = 1,  //!< unsigned integers
    halide_type_float = 2, //!< floating point numbers
    halide_type_handle = 3 //!< opaque pointer type (void *)
} halide_type_code_t;

// Note that while __attribute__ can go before or after the declaration,
// __declspec apparently is only allowed before.
#ifndef HALIDE_ATTRIBUTE_ALIGN
    #ifdef _MSC_VER
        #define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
    #else
        #define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
    #endif
#endif

/** A runtime tag for a type in the halide type system. Can be ints,
 * unsigned ints, or floats of various bit-widths (the 'bits'
 * field). Can also be vectors of the same (by setting the 'lanes'
 * field to something larger than one). This struct should be
 * exactly 32-bits in size. */
struct halide_type_t {
    /** The basic type code: signed integer, unsigned integer, or floating point. */
#if __cplusplus >= 201103L
    HALIDE_ATTRIBUTE_ALIGN(1) halide_type_code_t code; // halide_type_code_t
#else
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t code; // halide_type_code_t
#endif

    /** The number of bits of precision of a single scalar value of this type. */
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t bits;

    /** How many elements in a vector. This is 1 for scalar types. */
    HALIDE_ATTRIBUTE_ALIGN(2) uint16_t lanes;

#ifdef __cplusplus
    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    HALIDE_ALWAYS_INLINE halide_type_t(halide_type_code_t code, uint8_t bits, uint16_t lanes = 1)
        : code(code), bits(bits), lanes(lanes) {
    }

    /** Default constructor is required e.g. to declare halide_trace_event
     * instances. */
    HALIDE_ALWAYS_INLINE halide_type_t() : code((halide_type_code_t)0), bits(0), lanes(0) {}

    /** Compare two types for equality. */
    HALIDE_ALWAYS_INLINE bool operator==(const halide_type_t &other) const {
        return (code == other.code &&
                bits == other.bits &&
                lanes == other.lanes);
    }

    HALIDE_ALWAYS_INLINE bool operator!=(const halide_type_t &other) const {
        return !(*this == other);
    }

    /** Size in bytes for a single element, even if width is not 1, of this type. */
    HALIDE_ALWAYS_INLINE int64_t bytes() const { return (bits + 7) / 8; }
#endif
};

enum halide_trace_event_code_t {halide_trace_load = 0,
                                halide_trace_store = 1,
                                halide_trace_begin_realization = 2,
                                halide_trace_end_realization = 3,
                                halide_trace_produce = 4,
                                halide_trace_end_produce = 5,
                                halide_trace_consume = 6,
                                halide_trace_end_consume = 7,
                                halide_trace_begin_pipeline = 8,
                                halide_trace_end_pipeline = 9};

struct halide_trace_event_t {
    /** The name of the Func or Pipeline that this event refers to */
    const char *func;

    /** If the event type is a load or a store, this points to the
     * value being loaded or stored. Use the type field to safely cast
     * this to a concrete pointer type and retrieve it. For other
     * events this is null. */
    void *value;

    /** For loads and stores, an array which contains the location
     * being accessed. For vector loads or stores it is an array of
     * vectors of coordinates (the vector dimension is innermost).
     *
     * For realization or production-related events, this will contain
     * the mins and extents of the region being accessed, in the order
     * min0, extent0, min1, extent1, ...
     *
     * For pipeline-related events, this will be null.
     */
    int32_t *coordinates;

    /** If the event type is a load or a store, this is the type of
     * the data. Otherwise, the value is meaningless. */
    struct halide_type_t type;

    /** The type of event */
    enum halide_trace_event_code_t event;

    /* The ID of the parent event (see below for an explanation of
     * event ancestry). */
    int32_t parent_id;

    /** If this was a load or store of a Tuple-valued Func, this is
     * which tuple element was accessed. */
    int32_t value_index;

    /** The length of the coordinates array */
    int32_t dimensions;

#ifdef __cplusplus
    // If we don't explicitly mark the default ctor as inline,
    // certain build configurations can fail (notably iOS)
    HALIDE_ALWAYS_INLINE halide_trace_event_t() {}
#endif
};

/** Called when Funcs are marked as trace_load, trace_store, or
 * trace_realization. See Func::set_custom_trace. The default
 * implementation either prints events via halide_print, or if
 * HL_TRACE_FILE is defined, dumps the trace to that file in a
 * sequence of trace packets. The header for a trace packet is defined
 * below. If the trace is going to be large, you may want to make the
 * file a named pipe, and then read from that pipe into gzip.
 *
 * halide_trace returns a unique ID which will be passed to future
 * events that "belong" to the earlier event as the parent id. The
 * ownership hierarchy looks like:
 *
 * begin_pipeline
 * +--begin_realization
 * |  +--produce
 * |  |  +--load/store
 * |  |  +--end_produce
 * |  +--consume
 * |  |  +--load
 * |  |  +--end_consume
 * |  +--end_realization
 * +--end_pipeline
 *
 * Threading means that ownership cannot be inferred from the ordering
 * of events. There can be many active realizations of a given
 * function, or many active productions for a single
 * realization. Within a single production, the ordering of events is
 * meaningful.
 */
// @}
extern int32_t halide_trace(void *user_context, const struct halide_trace_event_t *event);
extern int32_t halide_default_trace(void *user_context, const struct halide_trace_event_t *event);
typedef int32_t (*halide_trace_t)(void *user_context, const struct halide_trace_event_t *);
extern halide_trace_t halide_set_custom_trace(halide_trace_t trace);
// @}

/** The header of a packet in a binary trace. All fields are 32-bit. */
struct halide_trace_packet_t {
    /** The total size of this packet in bytes. Always a multiple of
     * four. Equivalently, the number of bytes until the next
     * packet. */
    uint32_t size;

    /** The id of this packet (for the purpose of parent_id). */
    int32_t id;

    /** The remaining fields are equivalent to those in halide_trace_event_t */
    // @{
    struct halide_type_t type;
    enum halide_trace_event_code_t event;
    int32_t parent_id;
    int32_t value_index;
    int32_t dimensions;
    // @}

    #ifdef __cplusplus
    // If we don't explicitly mark the default ctor as inline,
    // certain build configurations can fail (notably iOS)
    HALIDE_ALWAYS_INLINE halide_trace_packet_t() {}

    /** Get the coordinates array, assuming this packet is laid out in
     * memory as it was written. The coordinates array comes
     * immediately after the packet header. */
    HALIDE_ALWAYS_INLINE const int *coordinates() const {
        return (const int *)(this + 1);
    }

    /** Get the value, assuming this packet is laid out in memory as
     * it was written. The packet comes immediately after the coordinates
     * array. */
    HALIDE_ALWAYS_INLINE const void *value() const {
        return (const void *)(coordinates() + dimensions);
    }

    /** Get the func name, assuming this packet is laid out in memory
     * as it was written. It comes after the value. */
    HALIDE_ALWAYS_INLINE const char *func() const {
        return (const char *)value() + type.lanes * type.bytes();
    }
    #endif
};



/** Set the file descriptor that Halide should write binary trace
 * events to. If called with 0 as the argument, Halide outputs trace
 * information to stdout in a human-readable format. If never called,
 * Halide checks the for existence of an environment variable called
 * HL_TRACE_FILE and opens that file. If HL_TRACE_FILE is not defined,
 * it outputs trace information to stdout in a human-readable
 * format. */
extern void halide_set_trace_file(int fd);

/** Halide calls this to retrieve the file descriptor to write binary
 * trace events to. The default implementation returns the value set
 * by halide_set_trace_file. Implement it yourself if you wish to use
 * a custom file descriptor per user_context. Return zero from your
 * implementation to tell Halide to print human-readable trace
 * information to stdout. */
extern int halide_get_trace_file(void *user_context);

/** If tracing is writing to a file. This call closes that file
 * (flushing the trace). Returns zero on success. */
extern int halide_shutdown_trace();

/** All Halide GPU or device backend implementations provide an
 * interface to be used with halide_device_malloc, etc. This is
 * accessed via the functions below.
 */

/** An opaque struct containing per-GPU API implementations of the
 * device functions. */
struct halide_device_interface_impl_t;

/** Each GPU API provides a halide_device_interface_t struct pointing
 * to the code that manages device allocations. You can access these
 * functions directly from the struct member function pointers, or by
 * calling the functions declared below. Note that the global
 * functions are not available when using Halide as a JIT compiler.
 * If you are using raw halide_buffer_t in that context you must use
 * the function pointers in the device_interface struct.
 *
 * The function pointers below are currently the same for every GPU
 * API; only the impl field varies. These top-level functions do the
 * bookkeeping that is common across all GPU APIs, and then dispatch
 * to more API-specific functions via another set of function pointers
 * hidden inside the impl field.
 */
struct halide_device_interface_t {
    int (*device_malloc)(void *user_context, struct halide_buffer_t *buf,
                         const struct halide_device_interface_t *device_interface);
    int (*device_free)(void *user_context, struct halide_buffer_t *buf);
    int (*device_sync)(void *user_context, struct halide_buffer_t *buf);
    void (*device_release)(void *user_context,
                          const struct halide_device_interface_t *device_interface);
    int (*copy_to_host)(void *user_context, struct halide_buffer_t *buf);
    int (*copy_to_device)(void *user_context, struct halide_buffer_t *buf,
                          const struct halide_device_interface_t *device_interface);
    int (*device_and_host_malloc)(void *user_context, struct halide_buffer_t *buf,
                                  const struct halide_device_interface_t *device_interface);
    int (*device_and_host_free)(void *user_context, struct halide_buffer_t *buf);
    int (*buffer_copy)(void *user_context, struct halide_buffer_t *src,
                       const struct halide_device_interface_t *dst_device_interface, struct halide_buffer_t *dst);
    int (*device_crop)(void *user_context, const struct halide_buffer_t *src,
                       struct halide_buffer_t *dst);
    int (*device_release_crop)(void *user_context, struct halide_buffer_t *buf);
    int (*wrap_native)(void *user_context, struct halide_buffer_t *buf, uint64_t handle,
                       const struct halide_device_interface_t *device_interface);
    int (*detach_native)(void *user_context, struct halide_buffer_t *buf);
    const struct halide_device_interface_impl_t *impl;
};

/** Release all data associated with the given device interface, in
 * particular all resources (memory, texture, context handles)
 * allocated by Halide. Must be called explicitly when using AOT
 * compilation. */
extern void halide_device_release(void *user_context,
                                  const struct halide_device_interface_t *device_interface);

/** Copy image data from device memory to host memory. This must be called
 * explicitly to copy back the results of a GPU-based filter. */
extern int halide_copy_to_host(void *user_context, struct halide_buffer_t *buf);

/** Copy image data from host memory to device memory. This should not
 * be called directly; Halide handles copying to the device
 * automatically.  If interface is NULL and the bug has a non-zero dev
 * field, the device associated with the dev handle will be
 * used. Otherwise if the dev field is 0 and interface is NULL, an
 * error is returned. */
extern int halide_copy_to_device(void *user_context, struct halide_buffer_t *buf,
                                 const struct halide_device_interface_t *device_interface);

/** Copy data from one buffer to another. The buffers may have
 * different shapes and sizes, but the destination buffer's shape must
 * be contained within the source buffer's shape. That is, for each
 * dimension, the min on the destination buffer must be greater than
 * or equal to the min on the source buffer, and min+extent on the
 * destination buffer must be less that or equal to min+extent on the
 * source buffer. The source data is pulled from either device or
 * host memory on the source, depending on the dirty flags. host is
 * preferred if both are valid. The dst_device_interface parameter
 * controls the destination memory space. NULL means host memory. */
extern int halide_buffer_copy(void *user_context, struct halide_buffer_t *src,
                              const struct halide_device_interface_t *dst_device_interface,
                              struct halide_buffer_t *dst);

/** Give the destination buffer a device allocation which is an alias
 * for the same coordinate range in the source buffer. Modifies the
 * device, device_interface, and the device_dirty flag only. Only
 * supported by some device APIs (others will return
 * halide_error_code_device_crop_unsupported). Call
 * halide_device_release_crop instead of halide_device_free to clean
 * up resources associated with the cropped view. Do not free the
 * device allocation on the source buffer while the destination buffer
 * still lives. Note that the two buffers do not share dirty flags, so
 * care must be taken to update them together as needed. Note also
 * that device interfaces which support cropping may still not support
 * cropping a crop. Instead, create a new crop of the parent
 * buffer. */
extern int halide_device_crop(void *user_context,
                              const struct halide_buffer_t *src,
                              struct halide_buffer_t *dst);

/** Release any resources associated with a cropped view of another
 * buffer. */
extern int halide_device_release_crop(void *user_context,
                                      struct halide_buffer_t *buf);

/** Wait for current GPU operations to complete. Calling this explicitly
 * should rarely be necessary, except maybe for profiling. */
extern int halide_device_sync(void *user_context, struct halide_buffer_t *buf);

/** Allocate device memory to back a halide_buffer_t. */
extern int halide_device_malloc(void *user_context, struct halide_buffer_t *buf,
                                const struct halide_device_interface_t *device_interface);

/** Free device memory. */
extern int halide_device_free(void *user_context, struct halide_buffer_t *buf);

/** Wrap or detach a native device handle, setting the device field
 * and device_interface field as appropriate for the given GPU
 * API. The meaning of the opaque handle is specific to the device
 * interface, so if you know the device interface in use, call the
 * more specific functions in the runtime headers for your specific
 * device API instead (e.g. HalideRuntimeCuda.h). */
// @{
extern int halide_device_wrap_native(void *user_context,
                                     struct halide_buffer_t *buf,
                                     uint64_t handle,
                                     const struct halide_device_interface_t *device_interface);
extern int halide_device_detach_native(void *user_context, struct halide_buffer_t *buf);
// @}

/** Versions of the above functions that accept legacy buffer_t structs. */
// @{
extern int halide_copy_to_host_legacy(void *user_context, struct buffer_t *buf);
extern int halide_copy_to_device_legacy(void *user_context, struct buffer_t *buf,
                                 const struct halide_device_interface_t *device_interface);
extern int halide_device_sync_legacy(void *user_context, struct buffer_t *buf);
extern int halide_device_malloc_legacy(void *user_context, struct buffer_t *buf,
                                const struct halide_device_interface_t *device_interface);
extern int halide_device_free_legacy(void *user_context, struct buffer_t *buf);
// @}

/** Selects which gpu device to use. 0 is usually the display
 * device. If never called, Halide uses the environment variable
 * HL_GPU_DEVICE. If that variable is unset, Halide uses the last
 * device. Set this to -1 to use the last device. */
extern void halide_set_gpu_device(int n);

/** Halide calls this to get the desired halide gpu device
 * setting. Implement this yourself to use a different gpu device per
 * user_context. The default implementation returns the value set by
 * halide_set_gpu_device, or the environment variable
 * HL_GPU_DEVICE. */
extern int halide_get_gpu_device(void *user_context);

/** Set the soft maximum amount of memory, in bytes, that the LRU
 *  cache will use to memoize Func results.  This is not a strict
 *  maximum in that concurrency and simultaneous use of memoized
 *  reults larger than the cache size can both cause it to
 *  temporariliy be larger than the size specified here.
 */
extern void halide_memoization_cache_set_size(int64_t size);

/** Given a cache key for a memoized result, currently constructed
 *  from the Func name and top-level Func name plus the arguments of
 *  the computation, determine if the result is in the cache and
 *  return it if so. (The internals of the cache key should be
 *  considered opaque by this function.) If this routine returns true,
 *  it is a cache miss. Otherwise, it will return false and the
 *  buffers passed in will be filled, via copying, with memoized
 *  data. The last argument is a list if halide_buffer_t pointers which
 *  represents the outputs of the memoized Func. If the Func does not
 *  return a Tuple, there will only be one halide_buffer_t in the list. The
 *  tuple_count parameters determines the length of the list.
 *
 * The return values are:
 * -1: Signals an error.
 *  0: Success and cache hit.
 *  1: Success and cache miss.
 */
extern int halide_memoization_cache_lookup(void *user_context, const uint8_t *cache_key, int32_t size,
                                           struct halide_buffer_t *realized_bounds,
                                           int32_t tuple_count, struct halide_buffer_t **tuple_buffers);

/** Given a cache key for a memoized result, currently constructed
 *  from the Func name and top-level Func name plus the arguments of
 *  the computation, store the result in the cache for futre access by
 *  halide_memoization_cache_lookup. (The internals of the cache key
 *  should be considered opaque by this function.) Data is copied out
 *  from the inputs and inputs are unmodified. The last argument is a
 *  list if halide_buffer_t pointers which represents the outputs of the
 *  memoized Func. If the Func does not return a Tuple, there will
 *  only be one halide_buffer_t in the list. The tuple_count parameters
 *  determines the length of the list.
 *
 * If there is a memory allocation failure, the store does not store
 * the data into the cache.
 */
extern int halide_memoization_cache_store(void *user_context, const uint8_t *cache_key, int32_t size,
                                          struct halide_buffer_t *realized_bounds,
                                          int32_t tuple_count,
                                          struct halide_buffer_t **tuple_buffers);

/** If halide_memoization_cache_lookup succeeds,
 * halide_memoization_cache_release must be called to signal the
 * storage is no longer being used by the caller. It will be passed
 * the host pointer of one the buffers returned by
 * halide_memoization_cache_lookup. That is
 * halide_memoization_cache_release will be called multiple times for
 * the case where halide_memoization_cache_lookup is handling multiple
 * buffers.  (This corresponds to memoizing a Tuple in Halide.) Note
 * that the host pointer must be sufficient to get to all information
 * the relase operation needs. The default Halide cache impleemntation
 * accomplishes this by storing extra data before the start of the user
 * modifiable host storage.
 *
 * This call is like free and does not have a failure return.
  */
extern void halide_memoization_cache_release(void *user_context, void *host);

/** Free all memory and resources associated with the memoization cache.
 * Must be called at a time when no other threads are accessing the cache.
 */
extern void halide_memoization_cache_cleanup();

/** Create a unique file with a name of the form prefixXXXXXsuffix in an arbitrary
 * (but writable) directory; this is typically $TMP or /tmp, but the specific
 * location is not guaranteed. (Note that the exact form of the file name
 * may vary; in particular, the suffix may be ignored on non-Posix systems.)
 * The file is created (but not opened), thus this can be called from
 * different threads (or processes, e.g. when building with parallel make)
 * without risking collision. Note that the caller is always responsible
 * for deleting this file. Returns nonzero value if an error occurs.
 */
extern int halide_create_temp_file(void *user_context,
  const char *prefix, const char *suffix,
  char *path_buf, size_t path_buf_size);

/** Annotate that a given range of memory has been initialized;
 * only used when Target::MSAN is enabled.
 *
 * The default implementation uses the LLVM-provided AnnotateMemoryIsInitialized() function.
 */
extern void halide_msan_annotate_memory_is_initialized(void *user_context, const void *ptr, uint64_t len);

/** Mark the data pointed to by the buffer_t as initialized (but *not* the buffer_t itself),
 * using halide_msan_annotate_memory_is_initialized() for marking.
 *
 * The default implementation takes pains to only mark the active memory ranges
 * (skipping padding), and sorting into ranges to always mark the smallest number of
 * ranges, in monotonically increasing memory order.
 *
 * Most client code should never need to replace the default implementation.
 */
extern void halide_msan_annotate_buffer_is_initialized(void *user_context, struct halide_buffer_t *buffer);
extern void halide_msan_annotate_buffer_is_initialized_as_destructor(void *user_context, void *buffer);

/** The error codes that may be returned by a Halide pipeline. */
enum halide_error_code_t {
    /** There was no error. This is the value returned by Halide on success. */
    halide_error_code_success = 0,

    /** An uncategorized error occurred. Refer to the string passed to halide_error. */
    halide_error_code_generic_error = -1,

    /** A Func was given an explicit bound via Func::bound, but this
     * was not large enough to encompass the region that is used of
     * the Func by the rest of the pipeline. */
    halide_error_code_explicit_bounds_too_small = -2,

    /** The elem_size field of a halide_buffer_t does not match the size in
     * bytes of the type of that ImageParam. Probable type mismatch. */
    halide_error_code_bad_type = -3,

    /** A pipeline would access memory outside of the halide_buffer_t passed
     * in. */
    halide_error_code_access_out_of_bounds = -4,

    /** A halide_buffer_t was given that spans more than 2GB of memory. */
    halide_error_code_buffer_allocation_too_large = -5,

    /** A halide_buffer_t was given with extents that multiply to a number
     * greater than 2^31-1 */
    halide_error_code_buffer_extents_too_large = -6,

    /** Applying explicit constraints on the size of an input or
     * output buffer shrank the size of that buffer below what will be
     * accessed by the pipeline. */
    halide_error_code_constraints_make_required_region_smaller = -7,

    /** A constraint on a size or stride of an input or output buffer
     * was not met by the halide_buffer_t passed in. */
    halide_error_code_constraint_violated = -8,

    /** A scalar parameter passed in was smaller than its minimum
     * declared value. */
    halide_error_code_param_too_small = -9,

    /** A scalar parameter passed in was greater than its minimum
     * declared value. */
    halide_error_code_param_too_large = -10,

    /** A call to halide_malloc returned NULL. */
    halide_error_code_out_of_memory = -11,

    /** A halide_buffer_t pointer passed in was NULL. */
    halide_error_code_buffer_argument_is_null = -12,

    /** debug_to_file failed to open or write to the specified
     * file. */
    halide_error_code_debug_to_file_failed = -13,

    /** The Halide runtime encountered an error while trying to copy
     * from device to host. Turn on -debug in your target string to
     * see more details. */
    halide_error_code_copy_to_host_failed = -14,

    /** The Halide runtime encountered an error while trying to copy
     * from host to device. Turn on -debug in your target string to
     * see more details. */
    halide_error_code_copy_to_device_failed = -15,

    /** The Halide runtime encountered an error while trying to
     * allocate memory on device. Turn on -debug in your target string
     * to see more details. */
    halide_error_code_device_malloc_failed = -16,

    /** The Halide runtime encountered an error while trying to
     * synchronize with a device. Turn on -debug in your target string
     * to see more details. */
    halide_error_code_device_sync_failed = -17,

    /** The Halide runtime encountered an error while trying to free a
     * device allocation. Turn on -debug in your target string to see
     * more details. */
    halide_error_code_device_free_failed = -18,

    /** Buffer has a non-zero device but no device interface, which
     * violates a Halide invariant. */
    halide_error_code_no_device_interface = -19,

    /** An error occurred when attempting to initialize the Matlab
     * runtime. */
    halide_error_code_matlab_init_failed = -20,

    /** The type of an mxArray did not match the expected type. */
    halide_error_code_matlab_bad_param_type = -21,

    /** There is a bug in the Halide compiler. */
    halide_error_code_internal_error = -22,

    /** The Halide runtime encountered an error while trying to launch
     * a GPU kernel. Turn on -debug in your target string to see more
     * details. */
    halide_error_code_device_run_failed = -23,

    /** The Halide runtime encountered a host pointer that violated
     * the alignment set for it by way of a call to
     * set_host_alignment */
    halide_error_code_unaligned_host_ptr = -24,

    /** A fold_storage directive was used on a dimension that is not
     * accessed in a monotonically increasing or decreasing fashion. */
    halide_error_code_bad_fold = -25,

    /** A fold_storage directive was used with a fold factor that was
     * too small to store all the values of a producer needed by the
     * consumer. */
    halide_error_code_fold_factor_too_small = -26,

    /** User-specified require() expression was not satisfied. */
    halide_error_code_requirement_failed = -27,

    /** At least one of the buffer's extents are negative. */
    halide_error_code_buffer_extents_negative = -28,

    /** A compiled pipeline was passed the old deprecated buffer_t
     * struct, and it could not be upgraded to a halide_buffer_t. */
    halide_error_code_failed_to_upgrade_buffer_t = -29,

    /** A compiled pipeline was passed the old deprecated buffer_t
     * struct in bounds inference mode, but the returned information
     * can't be expressed in the old buffer_t. */
    halide_error_code_failed_to_downgrade_buffer_t = -30,

    /** A specialize_fail() schedule branch was selected at runtime. */
    halide_error_code_specialize_fail = -31,

    /** The Halide runtime encountered an error while trying to wrap a
     * native device handle.  Turn on -debug in your target string to
     * see more details. */
    halide_error_code_device_wrap_native_failed = -32,

    /** The Halide runtime encountered an error while trying to detach
     * a native device handle.  Turn on -debug in your target string
     * to see more details. */
    halide_error_code_device_detach_native_failed = -33,

    /** The host field on an input or output was null, the device
     * field was not zero, and the pipeline tries to use the buffer on
     * the host. You may be passing a GPU-only buffer to a pipeline
     * which is scheduled to use it on the CPU. */
    halide_error_code_host_is_null = -34,

    /** A folded buffer was passed to an extern stage, but the region
     * touched wraps around the fold boundary. */
    halide_error_code_bad_extern_fold = -35,

    /** Buffer has a non-null device_interface but device is 0, which
     * violates a Halide invariant. */
    halide_error_code_device_interface_no_device= -36,

    /** Buffer has both host and device dirty bits set, which violates
     * a Halide invariant. */
    halide_error_code_host_and_device_dirty = -37,

    /** The halide_buffer_t * passed to a halide runtime routine is
     * nullptr and this is not allowed. */
    halide_error_code_buffer_is_null = -38,

    /** The Halide runtime encountered an error while trying to copy
     * from one buffer to another. Turn on -debug in your target
     * string to see more details. */
    halide_error_code_device_buffer_copy_failed = -39,

    /** Attempted to make cropped alias of a buffer with a device
     * field, but the device_interface does not support cropping. */
    halide_error_code_device_crop_unsupported = -40,

    /** Cropping a buffer failed for some other reason. Turn on -debug
     * in your target string. */
    halide_error_code_device_crop_failed = -41,

    /** An operation on a buffer required an allocation on a
     * particular device interface, but a device allocation already
     * existed on a different device interface. Free the old one
     * first. */
    halide_error_code_incompatible_device_interface = -42,
};

/** Halide calls the functions below on various error conditions. The
 * default implementations construct an error message, call
 * halide_error, then return the matching error code above. On
 * platforms that support weak linking, you can override these to
 * catch the errors individually. */

/** A call into an extern stage for the purposes of bounds inference
 * failed. Returns the error code given by the extern stage. */
extern int halide_error_bounds_inference_call_failed(void *user_context, const char *extern_stage_name, int result);

/** A call to an extern stage failed. Returned the error code given by
 * the extern stage. */
extern int halide_error_extern_stage_failed(void *user_context, const char *extern_stage_name, int result);

/** Various other error conditions. See the enum above for a
 * description of each. */
// @{
extern int halide_error_explicit_bounds_too_small(void *user_context, const char *func_name, const char *var_name,
                                                      int min_bound, int max_bound, int min_required, int max_required);
extern int halide_error_bad_type(void *user_context, const char *func_name,
                                 uint8_t code_given, uint8_t correct_code,
                                 uint8_t bits_given, uint8_t correct_bits,
                                 uint16_t lanes_given, uint16_t correct_lanes);
extern int halide_error_access_out_of_bounds(void *user_context, const char *func_name,
                                             int dimension, int min_touched, int max_touched,
                                             int min_valid, int max_valid);
extern int halide_error_buffer_allocation_too_large(void *user_context, const char *buffer_name,
                                                    uint64_t allocation_size, uint64_t max_size);
extern int halide_error_buffer_extents_negative(void *user_context, const char *buffer_name, int dimension, int extent);
extern int halide_error_buffer_extents_too_large(void *user_context, const char *buffer_name,
                                                 int64_t actual_size, int64_t max_size);
extern int halide_error_constraints_make_required_region_smaller(void *user_context, const char *buffer_name,
                                                                 int dimension,
                                                                 int constrained_min, int constrained_extent,
                                                                 int required_min, int required_extent);
extern int halide_error_constraint_violated(void *user_context, const char *var, int val,
                                            const char *constrained_var, int constrained_val);
extern int halide_error_param_too_small_i64(void *user_context, const char *param_name,
                                            int64_t val, int64_t min_val);
extern int halide_error_param_too_small_u64(void *user_context, const char *param_name,
                                            uint64_t val, uint64_t min_val);
extern int halide_error_param_too_small_f64(void *user_context, const char *param_name,
                                            double val, double min_val);
extern int halide_error_param_too_large_i64(void *user_context, const char *param_name,
                                            int64_t val, int64_t max_val);
extern int halide_error_param_too_large_u64(void *user_context, const char *param_name,
                                            uint64_t val, uint64_t max_val);
extern int halide_error_param_too_large_f64(void *user_context, const char *param_name,
                                            double val, double max_val);
extern int halide_error_out_of_memory(void *user_context);
extern int halide_error_buffer_argument_is_null(void *user_context, const char *buffer_name);
extern int halide_error_debug_to_file_failed(void *user_context, const char *func,
                                             const char *filename, int error_code);
extern int halide_error_unaligned_host_ptr(void *user_context, const char *func_name, int alignment);
extern int halide_error_host_is_null(void *user_context, const char *func_name);
extern int halide_error_failed_to_upgrade_buffer_t(void *user_context,
                                                   const char *input_name,
                                                   const char *reason);
extern int halide_error_failed_to_downgrade_buffer_t(void *user_context,
                                                     const char *input_name,
                                                     const char *reason);
extern int halide_error_bad_fold(void *user_context, const char *func_name, const char *var_name,
                                 const char *loop_name);
extern int halide_error_bad_extern_fold(void *user_context, const char *func_name,
                                        int dim, int min, int extent, int valid_min, int fold_factor);

extern int halide_error_fold_factor_too_small(void *user_context, const char *func_name, const char *var_name,
                                              int fold_factor, const char *loop_name, int required_extent);
extern int halide_error_requirement_failed(void *user_context, const char *condition, const char *message);
extern int halide_error_specialize_fail(void *user_context, const char *message);
extern int halide_error_no_device_interface(void *user_context);
extern int halide_error_device_interface_no_device(void *user_context);
extern int halide_error_host_and_device_dirty(void *user_context);
extern int halide_error_buffer_is_null(void *user_context, const char *routine);

// @}

/** Optional features a compilation Target can have.
 */
typedef enum halide_target_feature_t {
    halide_target_feature_jit = 0,  ///< Generate code that will run immediately inside the calling process.
    halide_target_feature_debug = 1,  ///< Turn on debug info and output for runtime code.
    halide_target_feature_no_asserts = 2,  ///< Disable all runtime checks, for slightly tighter code.
    halide_target_feature_no_bounds_query = 3, ///< Disable the bounds querying functionality.

    halide_target_feature_sse41 = 4,  ///< Use SSE 4.1 and earlier instructions. Only relevant on x86.
    halide_target_feature_avx = 5,  ///< Use AVX 1 instructions. Only relevant on x86.
    halide_target_feature_avx2 = 6,  ///< Use AVX 2 instructions. Only relevant on x86.
    halide_target_feature_fma = 7,  ///< Enable x86 FMA instruction
    halide_target_feature_fma4 = 8,  ///< Enable x86 (AMD) FMA4 instruction set
    halide_target_feature_f16c = 9,  ///< Enable x86 16-bit float support

    halide_target_feature_armv7s = 10,  ///< Generate code for ARMv7s. Only relevant for 32-bit ARM.
    halide_target_feature_no_neon = 11,  ///< Avoid using NEON instructions. Only relevant for 32-bit ARM.

    halide_target_feature_vsx = 12,  ///< Use VSX instructions. Only relevant on POWERPC.
    halide_target_feature_power_arch_2_07 = 13,  ///< Use POWER ISA 2.07 new instructions. Only relevant on POWERPC.

    halide_target_feature_cuda = 14,  ///< Enable the CUDA runtime. Defaults to compute capability 2.0 (Fermi)
    halide_target_feature_cuda_capability30 = 15,  ///< Enable CUDA compute capability 3.0 (Kepler)
    halide_target_feature_cuda_capability32 = 16,  ///< Enable CUDA compute capability 3.2 (Tegra K1)
    halide_target_feature_cuda_capability35 = 17,  ///< Enable CUDA compute capability 3.5 (Kepler)
    halide_target_feature_cuda_capability50 = 18,  ///< Enable CUDA compute capability 5.0 (Maxwell)

    halide_target_feature_opencl = 19,  ///< Enable the OpenCL runtime.
    halide_target_feature_cl_doubles = 20,  ///< Enable double support on OpenCL targets

    halide_target_feature_opengl = 21,  ///< Enable the OpenGL runtime.
    halide_target_feature_openglcompute = 22, ///< Enable OpenGL Compute runtime.

    halide_target_feature_unused_23 = 23, ///< Unused. (Formerly: Enable the RenderScript runtime.)

    halide_target_feature_user_context = 24,  ///< Generated code takes a user_context pointer as first argument

    halide_target_feature_matlab = 25,  ///< Generate a mexFunction compatible with Matlab mex libraries. See tools/mex_halide.m.

    halide_target_feature_profile = 26, ///< Launch a sampling profiler alongside the Halide pipeline that monitors and reports the runtime used by each Func
    halide_target_feature_no_runtime = 27, ///< Do not include a copy of the Halide runtime in any generated object file or assembly

    halide_target_feature_metal = 28, ///< Enable the (Apple) Metal runtime.
    halide_target_feature_mingw = 29, ///< For Windows compile to MinGW toolset rather then Visual Studio

    halide_target_feature_c_plus_plus_mangling = 30, ///< Generate C++ mangled names for result function, et al

    halide_target_feature_large_buffers = 31, ///< Enable 64-bit buffer indexing to support buffers > 2GB. Ignored if bits != 64.

    halide_target_feature_hvx_64 = 32, ///< Enable HVX 64 byte mode.
    halide_target_feature_hvx_128 = 33, ///< Enable HVX 128 byte mode.
    halide_target_feature_hvx_v62 = 34, ///< Enable Hexagon v62 architecture.
    halide_target_feature_fuzz_float_stores = 35, ///< On every floating point store, set the last bit of the mantissa to zero. Pipelines for which the output is very different with this feature enabled may also produce very different output on different processors.
    halide_target_feature_soft_float_abi = 36, ///< Enable soft float ABI. This only enables the soft float ABI calling convention, which does not necessarily use soft floats.
    halide_target_feature_msan = 37, ///< Enable hooks for MSAN support.
    halide_target_feature_avx512 = 38, ///< Enable the base AVX512 subset supported by all AVX512 architectures. The specific feature sets are AVX-512F and AVX512-CD. See https://en.wikipedia.org/wiki/AVX-512 for a description of each AVX subset.
    halide_target_feature_avx512_knl = 39, ///< Enable the AVX512 features supported by Knight's Landing chips, such as the Xeon Phi x200. This includes the base AVX512 set, and also AVX512-CD and AVX512-ER.
    halide_target_feature_avx512_skylake = 40, ///< Enable the AVX512 features supported by Skylake Xeon server processors. This adds AVX512-VL, AVX512-BW, and AVX512-DQ to the base set. The main difference from the base AVX512 set is better support for small integer ops. Note that this does not include the Knight's Landing features. Note also that these features are not available on Skylake desktop and mobile processors.
    halide_target_feature_avx512_cannonlake = 41, ///< Enable the AVX512 features expected to be supported by future Cannonlake processors. This includes all of the Skylake features, plus AVX512-IFMA and AVX512-VBMI.
    halide_target_feature_hvx_use_shared_object = 42, ///< Deprecated
    halide_target_feature_trace_loads = 43, ///< Trace all loads done by the pipeline. Equivalent to calling Func::trace_loads on every non-inlined Func.
    halide_target_feature_trace_stores = 44, ///< Trace all stores done by the pipeline. Equivalent to calling Func::trace_stores on every non-inlined Func.
    halide_target_feature_trace_realizations = 45, ///< Trace all realizations done by the pipeline. Equivalent to calling Func::trace_realizations on every non-inlined Func.
    halide_target_feature_cuda_capability61 = 46,  ///< Enable CUDA compute capability 6.1 (Pascal)
    halide_target_feature_hvx_v65 = 47, ///< Enable Hexagon v65 architecture.
    halide_target_feature_hvx_v66 = 48, ///< Enable Hexagon v66 architecture.
    halide_target_feature_end = 49, ///< A sentinel. Every target is considered to have this feature, and setting this feature does nothing.
} halide_target_feature_t;

/** This function is called internally by Halide in some situations to determine
 * if the current execution environment can support the given set of
 * halide_target_feature_t flags. The implementation must do the following:
 *
 * -- If there are flags set in features that the function knows *cannot* be supported, return 0.
 * -- Otherwise, return 1.
 * -- Note that any flags set in features that the function doesn't know how to test should be ignored;
 * this implies that a return value of 1 means "not known to be bad" rather than "known to be good".
 *
 * In other words: a return value of 0 means "It is not safe to use code compiled with these features",
 * while a return value of 1 means "It is not obviously unsafe to use code compiled with these features".
 *
 * The default implementation simply calls halide_default_can_use_target_features.
 */
// @{
extern int halide_can_use_target_features(uint64_t features);
typedef int (*halide_can_use_target_features_t)(uint64_t);
extern halide_can_use_target_features_t halide_set_custom_can_use_target_features(halide_can_use_target_features_t);
// @}

/**
 * This is the default implementation of halide_can_use_target_features; it is provided
 * for convenience of user code that may wish to extend halide_can_use_target_features
 * but continue providing existing support, e.g.
 *
 *     int halide_can_use_target_features(uint64_t features) {
 *          if (features & halide_target_somefeature) {
 *              if (!can_use_somefeature()) {
 *                  return 0;
 *              }
 *          }
 *          return halide_default_can_use_target_features(features);
 *     }
 */
extern int halide_default_can_use_target_features(uint64_t features);


typedef struct halide_dimension_t {
    int64_t min, extent, stride;

    // Per-dimension flags. None are defined yet (This is reserved for future use).
    uint32_t flags;

#ifdef __cplusplus
    HALIDE_ALWAYS_INLINE halide_dimension_t() : min(0), extent(0), stride(0), flags(0) {}
    HALIDE_ALWAYS_INLINE halide_dimension_t(int64_t m, int64_t e, int64_t s, uint64_t f = 0) :
        min(m), extent(e), stride(s), flags(f) {}

    HALIDE_ALWAYS_INLINE bool operator==(const halide_dimension_t &other) const {
        return (min == other.min) &&
            (extent == other.extent) &&
            (stride == other.stride) &&
            (flags == other.flags);
    }

    HALIDE_ALWAYS_INLINE bool operator!=(const halide_dimension_t &other) const {
        return !(*this == other);
    }
#endif
} halide_dimension_t;

#ifdef __cplusplus
} // extern "C"
#endif

typedef enum {halide_buffer_flag_host_dirty = 1,
              halide_buffer_flag_device_dirty = 2} halide_buffer_flags;

/**
 * The raw representation of an image passed around by generated
 * Halide code. It includes some stuff to track whether the image is
 * not actually in main memory, but instead on a device (like a
 * GPU). For a more convenient C++ wrapper, use Halide::Buffer<T>. */
typedef struct halide_buffer_t {
    /** A device-handle for e.g. GPU memory used to back this buffer. */
    uint64_t device;

    /** The interface used to interpret the above handle. */
    const struct halide_device_interface_t *device_interface;

    /** A pointer to the start of the data in main memory. In terms of
     * the Halide coordinate system, this is the address of the min
     * coordinates (defined below). */
    uint8_t* host;

    /** flags with various meanings. */
    uint64_t flags;

    /** The type of each buffer element. */
    struct halide_type_t type;

    /** The dimensionality of the buffer. */
    int32_t dimensions;

    /** The shape of the buffer. Halide does not own this array - you
     * must manage the memory for it yourself. */
    halide_dimension_t *dim;

    /** Pads the buffer up to a multiple of 8 bytes */
    void *padding;

#ifdef __cplusplus
    /** Convenience methods for accessing the flags */
    // @{
    HALIDE_ALWAYS_INLINE bool get_flag(halide_buffer_flags flag) const {
        return (flags & flag) != 0;
    }

    HALIDE_ALWAYS_INLINE void set_flag(halide_buffer_flags flag, bool value) {
        if (value) {
            flags |= flag;
        } else {
            flags &= ~flag;
        }
    }

    HALIDE_ALWAYS_INLINE bool host_dirty() const {
        return get_flag(halide_buffer_flag_host_dirty);
    }

    HALIDE_ALWAYS_INLINE bool device_dirty() const {
        return get_flag(halide_buffer_flag_device_dirty);
    }

    HALIDE_ALWAYS_INLINE void set_host_dirty(bool v = true) {
        set_flag(halide_buffer_flag_host_dirty, v);
    }

    HALIDE_ALWAYS_INLINE void set_device_dirty(bool v = true) {
        set_flag(halide_buffer_flag_device_dirty, v);
    }
    // @}

    /** The total number of elements this buffer represents. Equal to
     * the product of the extents */
    HALIDE_ALWAYS_INLINE size_t number_of_elements() const {
        size_t s = 1;
        for (int i = 0; i < dimensions; i++) {
            s *= dim[i].extent;
        }
        return s;
    }

    /** A pointer to the element with the lowest address. If all
     * strides are positive, equal to the host pointer. */
    HALIDE_ALWAYS_INLINE uint8_t *begin() const {
        ptrdiff_t index = 0;
        for (int i = 0; i < dimensions; i++) {
            if (dim[i].stride < 0) {
                index += dim[i].stride * (dim[i].extent - 1);
            }
        }
        return host + index * type.bytes();
    }

    /** A pointer to one beyond the element with the highest address. */
    HALIDE_ALWAYS_INLINE uint8_t *end() const {
        ptrdiff_t index = 0;
        for (int i = 0; i < dimensions; i++) {
            if (dim[i].stride > 0) {
                index += dim[i].stride * (dim[i].extent - 1);
            }
        }
        index += 1;
        return host + index * type.bytes();
    }

    /** The total number of bytes spanned by the data in memory. */
    HALIDE_ALWAYS_INLINE size_t size_in_bytes() const {
        return (size_t)(end() - begin());
    }

    /** A pointer to the element at the given location. */
    HALIDE_ALWAYS_INLINE uint8_t *address_of(const int *pos) const {
        ptrdiff_t index = 0;
        for (int i = 0; i < dimensions; i++) {
            index += dim[i].stride * (pos[i] - dim[i].min);
        }
        return host + index * type.bytes();
    }

    /** Attempt to call device_sync for the buffer. If the buffer
     * has no device_interface (or no device_sync), this is a quiet no-op.
     * Calling this explicitly should rarely be necessary, except for profiling. */
    HALIDE_ALWAYS_INLINE int device_sync(void *ctx = NULL) {
        if (device_interface && device_interface->device_sync) {
            return device_interface->device_sync(ctx, this);
        }
        return 0;
    }

    /** Check if an input buffer passed extern stage is a querying
     * bounds. Compared to doing the host pointer check directly,
     * this both adds clarity to code and will facilitate moving to
     * another representation for bounds query arguments. */
    HALIDE_ALWAYS_INLINE bool is_bounds_query() const {
        return host == NULL && device == 0;
    }

#endif
} halide_buffer_t;

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HALIDE_ATTRIBUTE_DEPRECATED
#ifdef HALIDE_ALLOW_DEPRECATED
#define HALIDE_ATTRIBUTE_DEPRECATED(x)
#else
#ifdef _MSC_VER
#define HALIDE_ATTRIBUTE_DEPRECATED(x) __declspec(deprecated(x))
#else
#define HALIDE_ATTRIBUTE_DEPRECATED(x) __attribute__((deprecated(x)))
#endif
#endif
#endif

/** The old buffer_t, included for compatibility with old code. Don't
 * use it. */
#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
typedef struct buffer_t {
    uint64_t dev;
    uint8_t* host;
    int32_t extent[4];
    int32_t stride[4];
    int32_t min[4];
    int32_t elem_size;
    HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
} buffer_t;
#endif // BUFFER_T_DEFINED

/** Copies host pointer, mins, extents, strides, and device state from
 * an old-style buffer_t into a new-style halide_buffer_t. The
 * dimensions and type fields of the new buffer_t should already be
 * set. Returns an error code if the upgrade could not be
 * performed. */
extern int halide_upgrade_buffer_t(void *user_context, const char *name,
                                   const buffer_t *old_buf, halide_buffer_t *new_buf);

/** Copies the host pointer, mins, extents, strides, and device state
 * from a halide_buffer_t to a buffer_t. Also sets elem_size. Useful
 * for backporting the results of bounds inference. */
extern int halide_downgrade_buffer_t(void *user_context, const char *name,
                                     const halide_buffer_t *new_buf, buffer_t *old_buf);

/** Copies the dirty flags and device allocation state from a new
 * buffer_t back to a legacy buffer_t. */
extern int halide_downgrade_buffer_t_device_fields(void *user_context, const char *name,
                                                   const halide_buffer_t *new_buf, buffer_t *old_buf);

/** halide_scalar_value_t is a simple union able to represent all the well-known
 * scalar values in a filter argument. Note that it isn't tagged with a type;
 * you must ensure you know the proper type before accessing. Most user
 * code will never need to create instances of this struct; its primary use
 * is to hold def/min/max values in a halide_filter_argument_t. (Note that
 * this is conceptually just a union; it's wrapped in a struct to ensure
 * that it doesn't get anonymized by LLVM.)
 */
struct halide_scalar_value_t {
    union {
        bool b;
        int8_t i8;
        int16_t i16;
        int32_t i32;
        int64_t i64;
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        float f32;
        double f64;
        void *handle;
    } u;
};

enum halide_argument_kind_t {
    halide_argument_kind_input_scalar = 0,
    halide_argument_kind_input_buffer = 1,
    halide_argument_kind_output_buffer = 2
};

/*
    These structs must be robust across different compilers and settings; when
    modifying them, strive for the following rules:

    1) All fields are explicitly sized. I.e. must use int32_t and not "int"
    2) All fields must land on an alignment boundary that is the same as their size
    3) Explicit padding is added to make that so
    4) The sizeof the struct is padded out to a multiple of the largest natural size thing in the struct
    5) don't forget that 32 and 64 bit pointers are different sizes
*/

/**
 * halide_filter_argument_t is essentially a plain-C-struct equivalent to
 * Halide::Argument; most user code will never need to create one.
 */
struct halide_filter_argument_t {
    const char *name;       // name of the argument; will never be null or empty.
    int32_t kind;           // actually halide_argument_kind_t
    int32_t dimensions;     // always zero for scalar arguments
    struct halide_type_t type;
    // These pointers should always be null for buffer arguments,
    // and *may* be null for scalar arguments. (A null value means
    // there is no def/min/max specified for this argument.)
    const struct halide_scalar_value_t *def;
    const struct halide_scalar_value_t *min;
    const struct halide_scalar_value_t *max;
};

struct halide_filter_metadata_t {
    /** version of this metadata; currently always 0. */
    int32_t version;

    /** The number of entries in the arguments field. This is always >= 1. */
    int32_t num_arguments;

    /** An array of the filters input and output arguments; this will never be
     * null. The order of arguments is not guaranteed (input and output arguments
     * may come in any order); however, it is guaranteed that all arguments
     * will have a unique name within a given filter. */
    const struct halide_filter_argument_t* arguments;

    /** The Target for which the filter was compiled. This is always
     * a canonical Target string (ie a product of Target::to_string). */
    const char* target;

    /** The function name of the filter. */
    const char* name;
};

/** The functions below here are relevant for pipelines compiled with
 * the -profile target flag, which runs a sampling profiler thread
 * alongside the pipeline. */

/** Per-Func state tracked by the sampling profiler. */
struct halide_profiler_func_stats {
    /** Total time taken evaluating this Func (in nanoseconds). */
    uint64_t time;

    /** The current memory allocation of this Func. */
    uint64_t memory_current;

    /** The peak memory allocation of this Func. */
    uint64_t memory_peak;

    /** The total memory allocation of this Func. */
    uint64_t memory_total;

    /** The peak stack allocation of this Func's threads. */
    uint64_t stack_peak;

    /** The average number of thread pool worker threads active while computing this Func. */
    uint64_t active_threads_numerator, active_threads_denominator;

    /** The name of this Func. A global constant string. */
    const char *name;

    /** The total number of memory allocation of this Func. */
    int num_allocs;
};

/** Per-pipeline state tracked by the sampling profiler. These exist
 * in a linked list. */
struct halide_profiler_pipeline_stats {
    /** Total time spent inside this pipeline (in nanoseconds) */
    uint64_t time;

    /** The current memory allocation of funcs in this pipeline. */
    uint64_t memory_current;

    /** The peak memory allocation of funcs in this pipeline. */
    uint64_t memory_peak;

    /** The total memory allocation of funcs in this pipeline. */
    uint64_t memory_total;

    /** The average number of thread pool worker threads doing useful
     * work while computing this pipeline. */
    uint64_t active_threads_numerator, active_threads_denominator;

    /** The name of this pipeline. A global constant string. */
    const char *name;

    /** An array containing states for each Func in this pipeline. */
    struct halide_profiler_func_stats *funcs;

    /** The next pipeline_stats pointer. It's a void * because types
     * in the Halide runtime may not currently be recursive. */
    void *next;

    /** The number of funcs in this pipeline. */
    int num_funcs;

    /** An internal base id used to identify the funcs in this pipeline. */
    int first_func_id;

    /** The number of times this pipeline has been run. */
    int runs;

    /** The total number of samples taken inside of this pipeline. */
    int samples;

    /** The total number of memory allocation of funcs in this pipeline. */
    int num_allocs;
};

/** The global state of the profiler. */
struct halide_profiler_state {
    /** Guards access to the fields below. If not locked, the sampling
     * profiler thread is free to modify things below (including
     * reordering the linked list of pipeline stats). */
    struct halide_mutex lock;

    /** The amount of time the profiler thread sleeps between samples
     * in milliseconds. Defaults to 1 */
    int sleep_time;

    /** An internal id used for bookkeeping. */
    int first_free_id;

    /** The id of the current running Func. Set by the pipeline, read
     * periodically by the profiler thread. */
    int current_func;

    /** The number of threads currently doing work. */
    int active_threads;

    /** A linked list of stats gathered for each pipeline. */
    struct halide_profiler_pipeline_stats *pipelines;

    /** Retrieve remote profiler state. Used so that the sampling
     * profiler can follow along with execution that occurs elsewhere,
     * e.g. on a DSP. If null, it reads from the int above instead. */
    void (*get_remote_profiler_state)(int *func, int *active_workers);

    /** Is the profiler thread running. */
    bool started;
};

/** Profiler func ids with special meanings. */
enum {
    /// current_func takes on this value when not inside Halide code
    halide_profiler_outside_of_halide = -1,
    /// Set current_func to this value to tell the profiling thread to
    /// halt. It will start up again next time you run a pipeline with
    /// profiling enabled.
    halide_profiler_please_stop = -2
};

/** Get a pointer to the global profiler state for programmatic
 * inspection. Lock it before using to pause the profiler. */
extern struct halide_profiler_state *halide_profiler_get_state();

/** Get a pointer to the pipeline state associated with pipeline_name.
 * This function grabs the global profiler state's lock on entry. */
extern struct halide_profiler_pipeline_stats *halide_profiler_get_pipeline_state(const char *pipeline_name);

/** Reset all profiler state.
 * WARNING: Do NOT call this method while any halide pipeline is
 * running; halide_profiler_memory_allocate/free and
 * halide_profiler_stack_peak_update update the profiler pipeline's
 * state without grabbing the global profiler state's lock. */
extern void halide_profiler_reset();

/** Print out timing statistics for everything run since the last
 * reset. Also happens at process exit. */
extern void halide_profiler_report(void *user_context);

/// \name "Float16" functions
/// These functions operate of bits (``uint16_t``) representing a half
/// precision floating point number (IEEE-754 2008 binary16).
//{@

/** Read bits representing a half precision floating point number and return
 *  the float that represents the same value */
extern float halide_float16_bits_to_float(uint16_t);

/** Read bits representing a half precision floating point number and return
 *  the double that represents the same value */
extern double halide_float16_bits_to_double(uint16_t);

// TODO: Conversion functions to half

//@}

#ifdef __cplusplus
} // End extern "C"
#endif

#ifdef __cplusplus

namespace {
template<typename T> struct check_is_pointer;
template<typename T> struct check_is_pointer<T *> {};
}

/** Construct the halide equivalent of a C type */
template<typename T>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of() {
    // Create a compile-time error if T is not a pointer (without
    // using any includes - this code goes into the runtime).
    check_is_pointer<T> check;
    (void)check;
    return halide_type_t(halide_type_handle, 64);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<float>() {
    return halide_type_t(halide_type_float, 32);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<double>() {
    return halide_type_t(halide_type_float, 64);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<bool>() {
    return halide_type_t(halide_type_uint, 1);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint8_t>() {
    return halide_type_t(halide_type_uint, 8);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint16_t>() {
    return halide_type_t(halide_type_uint, 16);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint32_t>() {
    return halide_type_t(halide_type_uint, 32);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<uint64_t>() {
    return halide_type_t(halide_type_uint, 64);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int8_t>() {
    return halide_type_t(halide_type_int, 8);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int16_t>() {
    return halide_type_t(halide_type_int, 16);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int32_t>() {
    return halide_type_t(halide_type_int, 32);
}

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<int64_t>() {
    return halide_type_t(halide_type_int, 64);
}

#endif

#endif // HALIDE_HALIDERUNTIME_H

namespace Halide {

/** Query whether Halide was compiled with exceptions. */
EXPORT bool exceptions_enabled();

/** A base class for Halide errors. */
struct Error : public std::runtime_error {
    // Give each class a non-inlined constructor so that the type
    // doesn't get separately instantiated in each compilation unit.
    EXPORT Error(const std::string &msg);
};

/** An error that occurs while running a JIT-compiled Halide pipeline. */
struct RuntimeError : public Error {
    EXPORT RuntimeError(const std::string &msg);
};

/** An error that occurs while compiling a Halide pipeline that Halide
 * attributes to a user error. */
struct CompileError : public Error {
    EXPORT CompileError(const std::string &msg);
};

/** An error that occurs while compiling a Halide pipeline that Halide
 * attributes to an internal compiler bug, or to an invalid use of
 * Halide's internals. */
struct InternalError : public Error {
    EXPORT InternalError(const std::string &msg);
};

/** CompileTimeErrorReporter is used at compile time (*not* runtime) when
 * an error or warning is generated by Halide. Note that error() is called
 * a fatal error has occurred, and returning to Halide may cause a crash;
 * implementations of CompileTimeErrorReporter::error() should never return.
 * (Implementations of CompileTimeErrorReporter::warning() may return but
 * may also abort(), exit(), etc.)
 */
class CompileTimeErrorReporter {
public:
    virtual ~CompileTimeErrorReporter() {}
    virtual void warning(const char* msg) = 0;
    virtual void error(const char* msg) = 0;
};

/** The default error reporter logs to stderr, then throws an exception
 * (if WITH_EXCEPTIONS) or calls abort (if not). This allows customization
 * of that behavior if a more gentle response to error reporting is desired.
 * Note that error_reporter is expected to remain valid across all Halide usage;
 * it is up to the caller to ensure that this is the case (and to do any
 * cleanup necessary).
 */
EXPORT void set_custom_compile_time_error_reporter(CompileTimeErrorReporter* error_reporter);

namespace Internal {

struct ErrorReport {
    enum {
        User = 0x0001,
        Warning = 0x0002,
        Runtime = 0x0004
    };

    std::ostringstream msg;
    const int flags;

    EXPORT ErrorReport(const char *f, int l, const char *cs, int flags);

    // Just a trick used to convert RValue into LValue
    HALIDE_ALWAYS_INLINE ErrorReport& ref() { return *this; }

    template<typename T>
    ErrorReport &operator<<(const T &x) {
        msg << x;
        return *this;
    }

    /** When you're done using << on the object, and let it fall out of
     * scope, this errors out, or throws an exception if they are
     * enabled. This is a little dangerous because the destructor will
     * also be called if there's an exception in flight due to an
     * error in one of the arguments passed to operator<<. We handle
     * this by only actually throwing if there isn't an exception in
     * flight already.
     */
#if __cplusplus >= 201100 || _MSC_VER >= 1900
    EXPORT ~ErrorReport() noexcept(false);
#else
    EXPORT ~ErrorReport();
#endif
};

// This uses operator precedence as a trick to avoid argument evaluation if
// an assertion is true: it is intended to be used as part of the
// _halide_internal_assertion macro, to coerce the result of the stream
// expression to void (to match the condition-is-false case).
class Voidifier {
 public:
  HALIDE_ALWAYS_INLINE Voidifier() {}
  // This has to be an operator with a precedence lower than << but
  // higher than ?:
  HALIDE_ALWAYS_INLINE void operator&(ErrorReport&) {}
};

/**
 * _halide_internal_assertion is used to implement our assertion macros
 * in such a way that the messages output for the assertion are only
 * evaluated if the assertion's value is false.
 *
 * Note that this macro intentionally has no parens internally; in actual
 * use, the implicit grouping will end up being
 *
 *   condition ? (void) : (Voidifier() & (ErrorReport << arg1 << arg2 ... << argN))
 *
 * This (regrettably) requires a macro to work, but has the highly desirable
 * effect that all assertion parameters are totally skipped (not ever evaluated)
 * when the assertion is true.
 */
#define _halide_internal_assertion(condition, flags) \
  (condition)                                        \
      ? (void)0                                      \
      : ::Halide::Internal::Voidifier() &            \
        ::Halide::Internal::ErrorReport(__FILE__, __LINE__, #condition, flags).ref()


#define internal_error            Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, 0)
#define user_error                Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, Halide::Internal::ErrorReport::User)
#define user_warning              Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, Halide::Internal::ErrorReport::User | Halide::Internal::ErrorReport::Warning)
#define halide_runtime_error      Halide::Internal::ErrorReport(__FILE__, __LINE__, nullptr, Halide::Internal::ErrorReport::User | Halide::Internal::ErrorReport::Runtime)

#define internal_assert(c)        _halide_internal_assertion(c, 0)
#define user_assert(c)            _halide_internal_assertion(c, Halide::Internal::ErrorReport::User)

// The nicely named versions get cleaned up at the end of Halide.h,
// but user code might want to do halide-style user_asserts (e.g. the
// Extern macros introduce calls to user_assert), so for that purpose
// we define an equivalent macro that can be used outside of Halide.h
#define _halide_user_assert(c)     _halide_internal_assertion(c, Halide::Internal::ErrorReport::User)

// N.B. Any function that might throw a user_assert or user_error may
// not be inlined into the user's code, or the line number will be
// misattributed to Halide.h. Either make such functions internal to
// libHalide, or mark them as NO_INLINE.

}

}

#endif
#ifndef HALIDE_EXPR_H
#define HALIDE_EXPR_H

/** \file
 * Base classes for Halide expressions (\ref Halide::Expr) and statements (\ref Halide::Internal::Stmt)
 */

#include <string>
#include <vector>

#ifndef HALIDE_FLOAT16_H
#define HALIDE_FLOAT16_H
#include <stdint.h>
#include <string>
#ifndef HALIDE_ROUNDING_MODE_H
#define HALIDE_ROUNDING_MODE_H
namespace Halide {

/** Rounding modes (IEEE754 2008 4.3 Rounding-direction attributes) */
enum class RoundingMode {
    TowardZero, ///< Round towards zero (IEEE754 2008 4.3.2)
    ToNearestTiesToEven, ///< Round to nearest, when there is a tie pick even integral significand (IEEE754 2008 4.3.1)
    ToNearestTiesToAway, ///< Round to nearest, when there is a tie pick value furthest away from zero (IEEE754 2008 4.3.1)
    TowardPositiveInfinity, ///< Round towards positive infinity (IEEE754 2008 4.3.2)
    TowardNegativeInfinity ///< Round towards negative infinity (IEEE754 2008 4.3.2)
};

}
#endif

namespace Halide {

/** Class that provides a type that implements half precision
 *  floating point (IEEE754 2008 binary16) in software.
 *
 *  This type is enforced to be 16-bits wide and maintains no state
 *  other than the raw IEEE754 binary16 bits so that it can passed
 *  to code that checks a type's size and used for buffer_t allocation.
 * */
struct float16_t {
    // NOTE: Do not use virtual methods here
    // it will change the size of this data type.

    /// \name Constructors
    /// @{

    /** Construct from a float using a particular rounding mode.
     *  A warning will be emitted if the result cannot be represented exactly
     *  and error will be raised if the conversion results in overflow.
     *
     *  \param value the input float
     *  \param roundingMode The rounding mode to use
     *
     */
    EXPORT explicit float16_t(float value, RoundingMode roundingMode=RoundingMode::ToNearestTiesToEven);

    /** Construct from a double using a particular rounding mode.
     *  A warning will be emitted if the result cannot be represented exactly
     *  and error will be raised if the conversion results in overflow.
     *
     *  \param value the input double
     *  \param roundingMode The rounding mode to use
     *
     */
    EXPORT explicit float16_t(double value, RoundingMode roundingMode=RoundingMode::ToNearestTiesToEven);

    /** Construct by parsing a string using a particular rounding mode.
     *  A warning will be emitted if the result cannot be represented exactly
     *  and error will be raised if the conversion results in overflow.
     *
     *  \param stringRepr the input string. The string maybe in C99 hex format
     *         (e.g. ``-0x1.000p-1``) or in a decimal (e.g.``-0.5``) format.
     *
     *  \param roundingMode The rounding mode to use
     *
     */
    EXPORT explicit float16_t(const char *stringRepr, RoundingMode roundingMode=RoundingMode::ToNearestTiesToEven);

    /** Construct a float16_t with the bits initialised to 0. This represents
     * positive zero.*/
    EXPORT float16_t();

    /// @}

    // Use explicit to avoid accidently raising the precision
    /** Cast to float */
    EXPORT explicit operator float() const;
    /** Cast to double */
    EXPORT explicit operator double() const;

    // Be explicit about how the copy constructor is expected to behave
    EXPORT float16_t(const float16_t&) = default;

    // Be explicit about how assignment is expected to behave
    EXPORT float16_t& operator=(const float16_t&) = default;

    /** \name Convenience "constructors"
     */
    /**@{*/

    /** Get a new float16_t that represents zero
     * \param positive if true then returns positive zero otherwise returns
     *        negative zero.
     */
    EXPORT static float16_t make_zero(bool positive);

    /** Get a new float16_t that represents infinity
     * \param positive if true then returns positive infinity otherwise returns
     *        negative infinity.
     */
    EXPORT static float16_t make_infinity(bool positive);

    /** Get a new float16_t that represents NaN (not a number) */
    EXPORT static float16_t make_nan();

    /** Get a new float16_t with the given raw bits
     *
     * \param bits The bits conformant to IEEE754 binary16
     */
    EXPORT static float16_t make_from_bits(uint16_t bits);

    /** Get a new float16_t from a signed integer.
     *  It is not provided as a constructor to avoid call ambiguity
     * */
    EXPORT static float16_t make_from_signed_int(int64_t value, RoundingMode roundingMode=RoundingMode::ToNearestTiesToEven);
    /**@}*/

    /**\name Arithmetic operators
     * These compute the result of an arithmetic operation
     * using a particular ``roundingMode`` and return a new float16_t
     * representing the result.
     *
     * Exceptions are ignored.
     */
    /**@{*/
    /** add */
    EXPORT float16_t add(float16_t rhs, RoundingMode roundingMode) const;
    /** subtract */
    EXPORT float16_t subtract(float16_t rhs, RoundingMode roundingMode) const;
    /** multiply */
    EXPORT float16_t multiply(float16_t rhs, RoundingMode roundingMode) const;
    /** divide */
    EXPORT float16_t divide(float16_t denominator, RoundingMode roundingMode) const;
    /** IEEE-754 2008 5.3.1 General operations - remainder **/
    EXPORT float16_t remainder(float16_t denominator) const;
    /** C fmod() */
    EXPORT float16_t mod(float16_t denominator, RoundingMode roudingMode) const;
    /**@}*/


    /** Return a new float16_t with a negated sign bit*/
    EXPORT float16_t operator-() const;

    /** \name Overloaded arithmetic operators for convenience
     * These operators assume RoundingMode::ToNearestTiesToEven rounding
     */
    /**@{*/
    EXPORT float16_t operator+(float16_t rhs) const;
    EXPORT float16_t operator-(float16_t rhs) const;
    EXPORT float16_t operator*(float16_t rhs) const;
    EXPORT float16_t operator/(float16_t rhs) const;
    /**@}*/

    /** \name Comparison operators */
    /**@{*/
    /** Equality */
    EXPORT bool operator==(float16_t rhs) const;
    /** Not equal */
    EXPORT bool operator!=(float16_t rhs) const { return !(*this == rhs); }
    /** Greater than */
    EXPORT bool operator>(float16_t rhs) const;
    /** Less than */
    EXPORT bool operator<(float16_t rhs) const;
    /** Greater than or equal to*/
    EXPORT bool operator>=(float16_t rhs) const { return (*this > rhs) || (*this == rhs); }
    /** Less than or equal to*/
    EXPORT bool operator<=(float16_t rhs) const { return (*this < rhs) || (*this == rhs); }
    /** \return true if and only if the float16_t and ``rhs`` are not ordered. E.g.
     * NaN and a normalised number
     */
    EXPORT bool are_unordered(float16_t rhs) const;
    /**@}*/

    /** \name String output methods */
    /**@{*/
    /** Return a string in the C99 hex format (e.g.\ ``-0x1.000p-1``) that
     * represents this float16_t precisely.
     */
    EXPORT std::string to_hex_string() const;
    /** Returns a string in a decimal scientific notation (e.g.\ ``-5.0E-1``)
     * that represents the closest decimal value to this float16_t precise to
     * the number of significant digits requested.
     *
     * \param significantDigits The number of significant digits to use. If
     *        set to ``0`` then string returned will have enough precision to
     *        construct the same float16_t when using
     *        RoundingMode::ToNearestTiesToEven
     */
    EXPORT std::string to_decimal_string(unsigned int significantDigits = 0) const;
    /**@}*/

    /** \name Properties */
    /*@{*/
    EXPORT bool is_nan() const;
    EXPORT bool is_infinity() const;
    EXPORT bool is_negative() const;
    EXPORT bool is_zero() const;
    /*@}*/

    /** Returns the bits that represent this float16_t.
     *
     *  An alternative method to access the bits is to cast a pointer
     *  to this instance as a pointer to a uint16_t.
     **/
    EXPORT uint16_t to_bits() const;

private:
    // The raw bits.
    // This must be the **ONLY** data member so that
    // this data type is 16-bits wide.
    uint16_t data;
};
}  // namespace Halide

template<>
HALIDE_ALWAYS_INLINE halide_type_t halide_type_of<Halide::float16_t>() {
    return halide_type_t(halide_type_float, 16);
}

#endif
#ifndef HALIDE_TYPE_H
#define HALIDE_TYPE_H

#include <stdint.h>

/** \file
 * Defines halide types
 */

/** A set of types to represent a C++ function signature. This allows
 * two things.  First, proper prototypes can be provided for Halide
 * generated functions, giving better compile time type
 * checking. Second, C++ name mangling can be done to provide link
 * time type checking for both Halide generated functions and calls
 * from Halide to external functions.
 *
 * These are intended to be constexpr producable, but we don't depend
 * on C++11 yet. In C++14, it is possible these will be replaced with
 * introspection/reflection facilities.
 *
 * halide_handle_traits has to go outside the Halide namespace due to template
 * resolution rules. TODO(zalman): Do all types need to be in global namespace?
 */
 //@{

/** A structure to represent the (unscoped) name of a C++ composite type for use
 * as a single argument (or return value) in a function signature.
 *
 * Currently does not support the restrict qualifier, references, or
 * r-value references.  These features cannot be used in extern
 * function calls from Halide or in the generated function from
 * Halide, but their applicability seems limited anyway.
 */
struct halide_cplusplus_type_name {
    /// An enum to indicate whether a C++ type is non-composite, a struct, class, or union
    enum CPPTypeType {
        Simple, ///< "int"
        Struct, ///< "struct Foo"
        Class,  ///< "class Foo"
        Union,  ///< "union Foo"
        Enum,   ///< "enum Foo"
    } cpp_type_type;  // Note: order is reflected in map_to_name table in CPlusPlusMangle.cpp

    std::string name;

    halide_cplusplus_type_name(CPPTypeType cpp_type_type, const std::string &name)
        : cpp_type_type(cpp_type_type), name(name) {
    }

    bool operator==(const halide_cplusplus_type_name &rhs) const {
         return cpp_type_type == rhs.cpp_type_type &&
                name == rhs.name;
    }

    bool operator!=(const halide_cplusplus_type_name &rhs) const {
        return !(*this == rhs);
    }

    bool operator<(const halide_cplusplus_type_name &rhs) const {
         return cpp_type_type < rhs.cpp_type_type ||
                (cpp_type_type == rhs.cpp_type_type &&
                 name < rhs.name);
    }
};

/** A structure to represent the fully scoped name of a C++ composite
 * type for use in generating function signatures that use that type.
 *
 * This is intended to be a constexpr usable type, but we don't depend
 * on C++11 yet. In C++14, it is possible this will be replaced with
 * introspection/reflection facilities.
 */
struct halide_handle_cplusplus_type {
    halide_cplusplus_type_name inner_name;
    std::vector<std::string> namespaces;
    std::vector<halide_cplusplus_type_name> enclosing_types;

    /// One set of modifiers on a type.
    /// The const/volatile/restrict propertises are "inside" the pointer property.
    enum Modifier : uint8_t {
        Const = 1 << 0,    ///< Bitmask flag for "const"
        Volatile = 1 << 1, ///< Bitmask flag for "volatile"
        Restrict = 1 << 2, ///< Bitmask flag for "restrict"
        Pointer = 1 << 3,  ///< Bitmask flag for a pointer "*"
    };

    /// Qualifiers and indirections on type. 0 is innermost.
    std::vector<uint8_t> cpp_type_modifiers;

    /// References are separate because they only occur at the outermost level.
    /// No modifiers are needed for references as they are not allowed to apply
    /// to the reference itself. (This isn't true for restrict, but that is a C++
    /// extension anyway.) If modifiers are needed, the last entry in the above
    /// array would be the modifers for the reference.
    enum ReferenceType : uint8_t {
        NotReference = 0,
        LValueReference = 1, // "&"
        RValueReference = 2, // "&&"
    };
    ReferenceType reference_type;

    halide_handle_cplusplus_type(const halide_cplusplus_type_name &inner_name,
                                 const std::vector<std::string> &namespaces = { },
                                 const std::vector<halide_cplusplus_type_name> &enclosing_types = { },
                                 const std::vector<uint8_t> &modifiers = { },
                                 ReferenceType reference_type = NotReference)
    : inner_name(inner_name),
      namespaces(namespaces),
      enclosing_types(enclosing_types),
      cpp_type_modifiers(modifiers),
      reference_type(reference_type) {}
};
//@}

template<typename T>
struct halide_c_type_to_name {
  static const bool known_type = false;
};

#define HALIDE_DECLARE_EXTERN_TYPE(TypeType, Type)                      \
    template<> struct halide_c_type_to_name<Type> {                     \
        static const bool known_type = true;                            \
        static halide_cplusplus_type_name name() {                      \
            return { halide_cplusplus_type_name::TypeType, #Type};      \
        }                                                               \
    }

#define HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(T)     HALIDE_DECLARE_EXTERN_TYPE(Simple, T)
#define HALIDE_DECLARE_EXTERN_STRUCT_TYPE(T)     HALIDE_DECLARE_EXTERN_TYPE(Struct, T)
#define HALIDE_DECLARE_EXTERN_CLASS_TYPE(T)      HALIDE_DECLARE_EXTERN_TYPE(Class, T)
#define HALIDE_DECLARE_EXTERN_UNION_TYPE(T)      HALIDE_DECLARE_EXTERN_TYPE(Union, T)

HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(bool);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(int8_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(uint8_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(int16_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(uint16_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(int32_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(uint32_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(int64_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(uint64_t);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(float);
HALIDE_DECLARE_EXTERN_SIMPLE_TYPE(double);
HALIDE_DECLARE_EXTERN_STRUCT_TYPE(buffer_t);
HALIDE_DECLARE_EXTERN_STRUCT_TYPE(halide_buffer_t);
HALIDE_DECLARE_EXTERN_STRUCT_TYPE(halide_dimension_t);
HALIDE_DECLARE_EXTERN_STRUCT_TYPE(halide_device_interface_t);
HALIDE_DECLARE_EXTERN_STRUCT_TYPE(halide_filter_metadata_t);

// You can make arbitrary user-defined types be "Known" using the
// macro above. This is useful for making Param<> arguments for
// Generators type safe. e.g.,
//
//    struct MyFunStruct { ... };
//
//    ...
//
//    HALIDE_DECLARE_EXTERN_STRUCT_TYPE(MyFunStruct);
//
//    ...
//
//    class MyGenerator : public Generator<MyGenerator> {
//       Param<const MyFunStruct *> my_struct_ptr;
//       ...
//    };


// Default case (should be only Unknown types, since we specialize for Known types below).
// We require that all unknown types be pointers, and translate them all to void*
// (preserving const-ness and volatile-ness).
template<typename T, bool KnownType>
struct halide_internal_handle_traits {
    static const halide_handle_cplusplus_type *type_info(bool is_ptr,
            halide_handle_cplusplus_type::ReferenceType ref_type) {
        static_assert(!KnownType, "Only unknown types handled here");
        internal_assert(is_ptr) << "Unknown types must be pointers";
        internal_assert(ref_type == halide_handle_cplusplus_type::NotReference) << "Unknown types must not be references";
        static const halide_handle_cplusplus_type the_info{
            {halide_cplusplus_type_name::Simple, "void"},
            {},
            {},
            {
                (uint8_t)(halide_handle_cplusplus_type::Pointer |
                    (std::is_const<T>::value ? halide_handle_cplusplus_type::Const : 0) |
                    (std::is_volatile<T>::value ? halide_handle_cplusplus_type::Volatile : 0))
            },
            halide_handle_cplusplus_type::NotReference
        };
        return &the_info;
    }
};

// Known types
template<typename T>
struct halide_internal_handle_traits<T, true> {

    static const halide_handle_cplusplus_type make_info(bool is_ptr,
                                                        halide_handle_cplusplus_type::ReferenceType ref_type) {
        halide_handle_cplusplus_type the_info = {
            halide_c_type_to_name<typename std::remove_cv<T>::type>::name(),
            {},
            {},
            {
                (uint8_t)((is_ptr ? halide_handle_cplusplus_type::Pointer : 0) |
                    (std::is_const<T>::value ? halide_handle_cplusplus_type::Const : 0) |
                    (std::is_volatile<T>::value ? halide_handle_cplusplus_type::Volatile : 0))
            },
            ref_type
        };
        // Pull off any namespaces
        the_info.inner_name.name =
            Halide::Internal::extract_namespaces(the_info.inner_name.name,
                                                 the_info.namespaces);
        return the_info;
    }

    static const halide_handle_cplusplus_type *type_info(bool is_ptr,
                                                         halide_handle_cplusplus_type::ReferenceType ref_type) {
        static const halide_handle_cplusplus_type the_info = make_info(is_ptr, ref_type);
        return &the_info;
    }
};

/** A type traits template to provide a halide_handle_cplusplus_type
 * value from a C++ type.
 *
 * Note the type represented is implicitly a pointer.
 *
 * A NULL pointer of type halide_handle_traits represents "void *".
 * This is chosen for compactness or representation as Type is a very
 * widely used data structure.
 */
template<typename T>
struct halide_handle_traits {
    // NULL here means "void *". This trait must return a pointer to a
    // global structure. I.e. it should never be freed.
    inline static const halide_handle_cplusplus_type *type_info() { return nullptr; }
};

template<typename T>
struct halide_handle_traits<T *> {
    inline static const halide_handle_cplusplus_type *type_info() {
        return halide_internal_handle_traits<T, halide_c_type_to_name<typename std::remove_cv<T>::type>::known_type>::type_info(true, halide_handle_cplusplus_type::NotReference);
     }
};

template<typename T>
struct halide_handle_traits<T &> {
    inline static const halide_handle_cplusplus_type *type_info() {
        return halide_internal_handle_traits<T, halide_c_type_to_name<typename std::remove_cv<T>::type>::known_type>::type_info(false, halide_handle_cplusplus_type::LValueReference);
    }
};

template<typename T>
struct halide_handle_traits<T &&> {
    inline static const halide_handle_cplusplus_type *type_info() {
        return halide_internal_handle_traits<T, halide_c_type_to_name<typename std::remove_cv<T>::type>::known_type>::type_info(false, halide_handle_cplusplus_type::RValueReference);
    }
};

template<>
struct halide_handle_traits<const char *> {
    inline static const halide_handle_cplusplus_type *type_info() {
        static const halide_handle_cplusplus_type the_info{
            halide_cplusplus_type_name(halide_cplusplus_type_name::Simple, "char"),
            {}, {}, { halide_handle_cplusplus_type::Pointer |
                      halide_handle_cplusplus_type::Const}};
        return &the_info;
    }
};

namespace Halide {

struct Expr;

/** Types in the halide type system. They can be ints, unsigned ints,
 * or floats of various bit-widths (the 'bits' field). They can also
 * be vectors of the same (by setting the 'lanes' field to something
 * larger than one). Front-end code shouldn't use vector
 * types. Instead vectorize a function. */
struct Type {
  private:
    halide_type_t type;

  public:
    /** Aliases for halide_type_code_t values for legacy compatibility
     * and to match the Halide internal C++ style. */
    // @{
    static const halide_type_code_t Int = halide_type_int;
    static const halide_type_code_t UInt = halide_type_uint;
    static const halide_type_code_t Float = halide_type_float;
    static const halide_type_code_t Handle = halide_type_handle;
    // @}

    /** The number of bytes required to store a single scalar value of this type. Ignores vector lanes. */
    int bytes() const {return (bits() + 7) / 8;}

    // Default ctor initializes everything to predictable-but-unlikely values
    Type() : type(Handle, 0, 0), handle_type(nullptr) {}


    /** Construct a runtime representation of a Halide type from:
     * code: The fundamental type from an enum.
     * bits: The bit size of one element.
     * lanes: The number of vector elements in the type. */
    Type(halide_type_code_t code, int bits, int lanes, const halide_handle_cplusplus_type *handle_type = nullptr)
        : type(code, (uint8_t)bits, (uint16_t)lanes), handle_type(handle_type) {
    }

    /** Trivial copy constructor. */
    Type(const Type &that) = default;

    /** Type is a wrapper around halide_type_t with more methods for use
     * inside the compiler. This simply constructs the wrapper around
     * the runtime value. */
    Type(const halide_type_t &that, const halide_handle_cplusplus_type *handle_type = nullptr)
         : type(that), handle_type(handle_type) {}

    /** Unwrap the runtime halide_type_t for use in runtime calls, etc.
     * Representation is exactly equivalent. */
    operator halide_type_t() const { return type; }

    /** Return the underlying data type of an element as an enum value. */
    halide_type_code_t code() const { return (halide_type_code_t)type.code; }

    /** Return the bit size of a single element of this type. */
    int bits() const { return type.bits; }

    /** Return the number of vector elements in this type. */
    int lanes() const { return type.lanes; }

    /** Return Type with same number of bits and lanes, but new_code for a type code. */
    Type with_code(halide_type_code_t new_code) const {
        return Type(new_code, bits(), lanes(),
                    (new_code == code()) ? handle_type : nullptr);
    }

    /** Return Type with same type code and lanes, but new_bits for the number of bits. */
    Type with_bits(int new_bits) const {
        return Type(code(), new_bits, lanes(),
                    (new_bits == bits()) ? handle_type : nullptr);
    }

    /** Return Type with same type code and number of bits,
     * but new_lanes for the number of vector lanes. */
    Type with_lanes(int new_lanes) const {
        return Type(code(), bits(), new_lanes, handle_type);
    }

    /** Type to be printed when declaring handles of this type. */
    const halide_handle_cplusplus_type *handle_type;

    /** Is this type boolean (represented as UInt(1))? */
    bool is_bool() const {return code() == UInt && bits() == 1;}

    /** Is this type a vector type? (lanes() != 1).
     * TODO(abadams): Decide what to do for lanes() == 0. */
    bool is_vector() const {return lanes() != 1;}

    /** Is this type a scalar type? (lanes() == 1).
     * TODO(abadams): Decide what to do for lanes() == 0. */
    bool is_scalar() const {return lanes() == 1;}

    /** Is this type a floating point type (float or double). */
    bool is_float() const {return code() == Float;}

    /** Is this type a signed integer type? */
    bool is_int() const {return code() == Int;}

    /** Is this type an unsigned integer type? */
    bool is_uint() const {return code() == UInt;}

    /** Is this type an opaque handle type (void *) */
    bool is_handle() const {return code() == Handle;}

    /** Check that the type name of two handles matches. */
    EXPORT bool same_handle_type(const Type &other) const;

    /** Compare two types for equality */
    bool operator==(const Type &other) const {
        return code() == other.code() && bits() == other.bits() && lanes() == other.lanes() &&
            (code() != Handle || same_handle_type(other));
    }

    /** Compare two types for inequality */
    bool operator!=(const Type &other) const {
        return code() != other.code() || bits() != other.bits() || lanes() != other.lanes() ||
            (code() == Handle && !same_handle_type(other));
    }

    /** Compare ordering of two types so they can be used in certain containers and algorithms */
    bool operator<(const Type &other) const {
        return code() < other.code() || (code() == other.code() &&
              (bits() < other.bits() || (bits() == other.bits() &&
              (lanes() < other.lanes() || (lanes() == other.lanes() &&
              (code() == Handle && handle_type < other.handle_type))))));
    }

    /** Produce the scalar type (that of a single element) of this vector type */
    Type element_of() const {
        return with_lanes(1);
    }

    /** Can this type represent all values of another type? */
    EXPORT bool can_represent(Type other) const;

    /** Can this type represent a particular constant? */
    // @{
    EXPORT bool can_represent(double x) const;
    EXPORT bool can_represent(int64_t x) const;
    EXPORT bool can_represent(uint64_t x) const;
    // @}

    /** Check if an integer constant value is the maximum or minimum
     * representable value for this type. */
    // @{
    EXPORT bool is_max(uint64_t) const;
    EXPORT bool is_max(int64_t) const;
    EXPORT bool is_min(uint64_t) const;
    EXPORT bool is_min(int64_t) const;
    // @}

    /** Return an expression which is the maximum value of this type.
     * Returns infinity for types which can represent it. */
    EXPORT Expr max() const;

    /** Return an expression which is the minimum value of this type.
     * Returns -infinity for types which can represent it. */
    EXPORT Expr min() const;
};

/** Constructing a signed integer type */
inline Type Int(int bits, int lanes = 1) {
    return Type(Type::Int, bits, lanes);
}

/** Constructing an unsigned integer type */
inline Type UInt(int bits, int lanes = 1) {
    return Type(Type::UInt, bits, lanes);
}

/** Construct a floating-point type */
inline Type Float(int bits, int lanes = 1) {
    return Type(Type::Float, bits, lanes);
}

/** Construct a boolean type */
inline Type Bool(int lanes = 1) {
    return UInt(1, lanes);
}

/** Construct a handle type */
inline Type Handle(int lanes = 1, const halide_handle_cplusplus_type *handle_type = nullptr) {
    return Type(Type::Handle, 64, lanes, handle_type);
}

/** Construct the halide equivalent of a C type */
template<typename T>
inline Type type_of() {
    return Type(halide_type_of<T>(), halide_handle_traits<T>::type_info());
}

}  // namespace Halide

#endif
#ifndef HALIDE_INTRUSIVE_PTR_H
#define HALIDE_INTRUSIVE_PTR_H

/** \file
 *
 * Support classes for reference-counting via intrusive shared
 * pointers.
 */

#include <stdlib.h>
#include <atomic>


namespace Halide {
namespace Internal {

/** A class representing a reference count to be used with IntrusivePtr */
class RefCount {
    std::atomic<int> count;
public:
    RefCount() : count(0) {}
    int increment() {return ++count;} // Increment and return new value
    int decrement() {return --count;} // Decrement and return new value
    bool is_zero() const {return count == 0;}
};

/**
 * Because in this header we don't yet know how client classes store
 * their RefCount (and we don't want to depend on the declarations of
 * the client classes), any class that you want to hold onto via one
 * of these must provide implementations of ref_count and destroy,
 * which we forward-declare here.
 *
 * E.g. if you want to use IntrusivePtr<MyClass>, then you should
 * define something like this in MyClass.cpp (assuming MyClass has
 * a field: mutable RefCount ref_count):
 *
 * template<> RefCount &ref_count<MyClass>(const MyClass *c) {return c->ref_count;}
 * template<> void destroy<MyClass>(const MyClass *c) {delete c;}
 */
// @{
template<typename T> EXPORT RefCount &ref_count(const T *t);
template<typename T> EXPORT void destroy(const T *t);
// @}

/** Intrusive shared pointers have a reference count (a
 * RefCount object) stored in the class itself. This is perhaps more
 * efficient than storing it externally, but more importantly, it
 * means it's possible to recover a reference-counted handle from the
 * raw pointer, and it's impossible to have two different reference
 * counts attached to the same raw object. Seeing as we pass around
 * raw pointers to concrete IRNodes and Expr's interchangeably, this
 * is a useful property.
 */
template<typename T>
struct IntrusivePtr {
private:

    void incref(T *p) {
        if (p) {
            ref_count(p).increment();
        }
    };

    void decref(T *p) {
        if (p) {
            // Note that if the refcount is already zero, then we're
            // in a recursive destructor due to a self-reference (a
            // cycle), where the ref_count has been adjusted to remove
            // the counts due to the cycle. The next line then makes
            // the ref_count negative, which prevents actually
            // entering the destructor recursively.
            if (ref_count(p).decrement() == 0) {
                destroy(p);
            }
        }
    }

protected:
    T *ptr;

public:
    /** Access the raw pointer in a variety of ways.
     * Note that a "const IntrusivePtr<T>" is not the same thing as an
     * IntrusivePtr<const T>. So the methods that return the ptr are
     * const, despite not adding an extra const to T. */
    // @{
    T *get() const {
        return ptr;
    }

    T &operator*() const {
        return *ptr;
    }

    T *operator->() const {
        return ptr;
    }
    // @}

    ~IntrusivePtr() {
        decref(ptr);
    }

    IntrusivePtr() : ptr(nullptr) {
    }

    IntrusivePtr(T *p) : ptr(p) {
        incref(ptr);
    }

    IntrusivePtr(const IntrusivePtr<T> &other) : ptr(other.ptr) {
        incref(ptr);
    }

    IntrusivePtr(IntrusivePtr<T> &&other) : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    IntrusivePtr<T> &operator=(const IntrusivePtr<T> &other) {
        if (other.ptr == ptr) return *this;
        // Other can be inside of something owned by this, so we
        // should be careful to incref other before we decref
        // ourselves.
        T *temp = other.ptr;
        incref(temp);
        decref(ptr);
        ptr = temp;
        return *this;
    }

    IntrusivePtr<T> &operator=(IntrusivePtr<T> &&other) {
        std::swap(ptr, other.ptr);
        return *this;
    }

    /* Handles can be null. This checks that. */
    bool defined() const {
        return ptr != nullptr;
    }

    /* Check if two handles point to the same ptr. This is
     * equality of reference, not equality of value. */
    bool same_as(const IntrusivePtr &other) const {
        return ptr == other.ptr;
    }

    bool operator <(const IntrusivePtr<T> &other) const {
        return ptr < other.ptr;
    }

};

}
}

#endif

namespace Halide {
namespace Internal {

class IRVisitor;

/** All our IR node types get unique IDs for the purposes of RTTI */
enum class IRNodeType {
    IntImm,
    UIntImm,
    FloatImm,
    StringImm,
    Cast,
    Variable,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    And,
    Or,
    Not,
    Select,
    Load,
    Ramp,
    Broadcast,
    Call,
    Let,
    LetStmt,
    AssertStmt,
    ProducerConsumer,
    For,
    Store,
    Provide,
    Allocate,
    Free,
    Realize,
    Block,
    IfThenElse,
    Evaluate,
    Shuffle,
    Prefetch,
};

/** The abstract base classes for a node in the Halide IR. */
struct IRNode {

    /** We use the visitor pattern to traverse IR nodes throughout the
     * compiler, so we have a virtual accept method which accepts
     * visitors.
     */
    virtual void accept(IRVisitor *v) const = 0;
    IRNode(IRNodeType t) : node_type(t) {}
    virtual ~IRNode() {}

    /** These classes are all managed with intrusive reference
     * counting, so we also track a reference count. It's mutable
     * so that we can do reference counting even through const
     * references to IR nodes.
     */
    mutable RefCount ref_count;

    /** Each IR node subclass has a unique identifier. We can compare
     * these values to do runtime type identification. We don't
     * compile with rtti because that injects run-time type
     * identification stuff everywhere (and often breaks when linking
     * external libraries compiled without it), and we only want it
     * for IR nodes. One might want to put this value in the vtable,
     * but that adds another level of indirection, and for Exprs we
     * have 32 free bits in between the ref count and the Type
     * anyway, so this doesn't increase the memory footprint of an IR node.
     */
    IRNodeType node_type;
};

template<>
EXPORT inline RefCount &ref_count<IRNode>(const IRNode *n) {return n->ref_count;}

template<>
EXPORT inline void destroy<IRNode>(const IRNode *n) {delete n;}

/** IR nodes are split into expressions and statements. These are
   similar to expressions and statements in C - expressions
   represent some value and have some type (e.g. x + 3), and
   statements are side-effecting pieces of code that do not
   represent a value (e.g. assert(x > 3)) */

/** A base class for statement nodes. They have no properties or
   methods beyond base IR nodes for now. */
struct BaseStmtNode : public IRNode {
    BaseStmtNode(IRNodeType t) : IRNode(t) {}
};

/** A base class for expression nodes. They all contain their types
 * (e.g. Int(32), Float(32)) */
struct BaseExprNode : public IRNode {
    BaseExprNode(IRNodeType t) : IRNode(t) {}
    Type type;
};

/** We use the "curiously recurring template pattern" to avoid
   duplicated code in the IR Nodes. These classes live between the
   abstract base classes and the actual IR Nodes in the
   inheritance hierarchy. It provides an implementation of the
   accept function necessary for the visitor pattern to work, and
   a concrete instantiation of a unique IRNodeType per class. */
template<typename T>
struct ExprNode : public BaseExprNode {
    EXPORT void accept(IRVisitor *v) const;
    ExprNode() : BaseExprNode(T::_node_type) {}
    virtual ~ExprNode() {}
};

template<typename T>
struct StmtNode : public BaseStmtNode {
    EXPORT void accept(IRVisitor *v) const;
    StmtNode() : BaseStmtNode(T::_node_type) {}
    virtual ~StmtNode() {}
};

/** IR nodes are passed around opaque handles to them. This is a
   base class for those handles. It manages the reference count,
   and dispatches visitors. */
struct IRHandle : public IntrusivePtr<const IRNode> {
    IRHandle() : IntrusivePtr<const IRNode>() {}
    IRHandle(const IRNode *p) : IntrusivePtr<const IRNode>(p) {}

    /** Dispatch to the correct visitor method for this node. E.g. if
     * this node is actually an Add node, then this will call
     * IRVisitor::visit(const Add *) */
    void accept(IRVisitor *v) const {
        ptr->accept(v);
    }

    /** Downcast this ir node to its actual type (e.g. Add, or
     * Select). This returns nullptr if the node is not of the requested
     * type. Example usage:
     *
     * if (const Add *add = node->as<Add>()) {
     *   // This is an add node
     * }
     */
    template<typename T> const T *as() const {
        if (ptr && ptr->node_type == T::_node_type) {
            return (const T *)ptr;
        }
        return nullptr;
    }
};


/** Integer constants */
struct IntImm : public ExprNode<IntImm> {
    int64_t value;

    static const IntImm *make(Type t, int64_t value) {
        internal_assert(t.is_int() && t.is_scalar())
            << "IntImm must be a scalar Int\n";
        internal_assert(t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64)
            << "IntImm must be 8, 16, 32, or 64-bit\n";

        // Normalize the value by dropping the high bits
        value <<= (64 - t.bits());
        // Then sign-extending to get them back
        value >>= (64 - t.bits());

        IntImm *node = new IntImm;
        node->type = t;
        node->value = value;
        return node;
    }

    static const IRNodeType _node_type = IRNodeType::IntImm;
};

/** Unsigned integer constants */
struct UIntImm : public ExprNode<UIntImm> {
    uint64_t value;

    static const UIntImm *make(Type t, uint64_t value) {
        internal_assert(t.is_uint() && t.is_scalar())
            << "UIntImm must be a scalar UInt\n";
        internal_assert(t.bits() == 1 || t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64)
            << "UIntImm must be 1, 8, 16, 32, or 64-bit\n";

        // Normalize the value by dropping the high bits
        value <<= (64 - t.bits());
        value >>= (64 - t.bits());

        UIntImm *node = new UIntImm;
        node->type = t;
        node->value = value;
        return node;
    }

    static const IRNodeType _node_type = IRNodeType::UIntImm;
};

/** Floating point constants */
struct FloatImm : public ExprNode<FloatImm> {
    double value;

    static const FloatImm *make(Type t, double value) {
        internal_assert(t.is_float() && t.is_scalar())
            << "FloatImm must be a scalar Float\n";
        FloatImm *node = new FloatImm;
        node->type = t;
        switch (t.bits()) {
        case 16:
            node->value = (double)((float16_t)value);
            break;
        case 32:
            node->value = (float)value;
            break;
        case 64:
            node->value = value;
            break;
        default:
            internal_error << "FloatImm must be 16, 32, or 64-bit\n";
        }

        return node;
    }

    static const IRNodeType _node_type = IRNodeType::FloatImm;
};

/** String constants */
struct StringImm : public ExprNode<StringImm> {
    std::string value;

    static const StringImm *make(const std::string &val) {
        StringImm *node = new StringImm;
        node->type = type_of<const char *>();
        node->value = val;
        return node;
    }

    static const IRNodeType _node_type = IRNodeType::StringImm;
};

}  // namespace Internal

/** A fragment of Halide syntax. It's implemented as reference-counted
 * handle to a concrete expression node, but it's immutable, so you
 * can treat it as a value type. */
struct Expr : public Internal::IRHandle {
    /** Make an undefined expression */
    Expr() : Internal::IRHandle() {}

    /** Make an expression from a concrete expression node pointer (e.g. Add) */
    Expr(const Internal::BaseExprNode *n) : IRHandle(n) {}

    /** Make an expression representing numeric constants of various types. */
    // @{
    EXPORT explicit Expr(int8_t x)    : IRHandle(Internal::IntImm::make(Int(8), x)) {}
    EXPORT explicit Expr(int16_t x)   : IRHandle(Internal::IntImm::make(Int(16), x)) {}
    EXPORT          Expr(int32_t x)   : IRHandle(Internal::IntImm::make(Int(32), x)) {}
    EXPORT explicit Expr(int64_t x)   : IRHandle(Internal::IntImm::make(Int(64), x)) {}
    EXPORT explicit Expr(uint8_t x)   : IRHandle(Internal::UIntImm::make(UInt(8), x)) {}
    EXPORT explicit Expr(uint16_t x)  : IRHandle(Internal::UIntImm::make(UInt(16), x)) {}
    EXPORT explicit Expr(uint32_t x)  : IRHandle(Internal::UIntImm::make(UInt(32), x)) {}
    EXPORT explicit Expr(uint64_t x)  : IRHandle(Internal::UIntImm::make(UInt(64), x)) {}
    EXPORT          Expr(float16_t x) : IRHandle(Internal::FloatImm::make(Float(16), (double)x)) {}
    EXPORT          Expr(float x)     : IRHandle(Internal::FloatImm::make(Float(32), x)) {}
    EXPORT explicit Expr(double x)    : IRHandle(Internal::FloatImm::make(Float(64), x)) {}
    // @}

    /** Make an expression representing a const string (i.e. a StringImm) */
    EXPORT          Expr(const std::string &s) : IRHandle(Internal::StringImm::make(s)) {}

    /** Get the type of this expression node */
    Type type() const {
        return ((const Internal::BaseExprNode *)ptr)->type;
    }
};

/** This lets you use an Expr as a key in a map of the form
 * map<Expr, Foo, ExprCompare> */
struct ExprCompare {
    bool operator()(const Expr &a, const Expr &b) const {
        return a.get() < b.get();
    }
};

/** An enum describing a type of device API. Used by schedules, and in
 * the For loop IR node. */
enum class DeviceAPI {
    None, /// Used to denote for loops that run on the same device as the containing code.
    Host,
    Default_GPU,
    CUDA,
    OpenCL,
    GLSL,
    OpenGLCompute,
    Metal,
    Hexagon
};

/** An array containing all the device apis. Useful for iterating
 * through them. */
const DeviceAPI all_device_apis[] = {DeviceAPI::None,
                                     DeviceAPI::Host,
                                     DeviceAPI::Default_GPU,
                                     DeviceAPI::CUDA,
                                     DeviceAPI::OpenCL,
                                     DeviceAPI::GLSL,
                                     DeviceAPI::OpenGLCompute,
                                     DeviceAPI::Metal,
                                     DeviceAPI::Hexagon};

namespace Internal {

/** An enum describing a type of loop traversal. Used in schedules, and in
 * the For loop IR node. GPUBlock and GPUThread are implicitly parallel */
enum class ForType {
    Serial,
    Parallel,
    Vectorized,
    Unrolled,
    GPUBlock,
    GPUThread
};


/** A reference-counted handle to a statement node. */
struct Stmt : public IRHandle {
    Stmt() : IRHandle() {}
    Stmt(const BaseStmtNode *n) : IRHandle(n) {}

    /** This lets you use a Stmt as a key in a map of the form
     * map<Stmt, Foo, Stmt::Compare> */
    struct Compare {
        bool operator()(const Stmt &a, const Stmt &b) const {
            return a.ptr < b.ptr;
        }
    };
};


}  // namespace Internal
}  // namespace Halide

#endif
#ifndef HALIDE_FUNCTION_H
#define HALIDE_FUNCTION_H

/** \file
 * Defines the internal representation of a halide function and related classes
 */

#ifndef HALIDE_FUNCTION_PTR_H
#define HALIDE_FUNCTION_PTR_H


namespace Halide {
namespace Internal {

/** Functions are allocated in groups for memory management. Each
 * group has a ref count associated with it. All within-group
 * references must be weak. If there are any references from outside
 * the group, at least one must be strong.  Within-group references
 * may form cycles, but there may not be reference cycles that span
 * multiple groups. These rules are not enforced automatically. */
struct FunctionGroup;

/** The opaque struct describing a Halide function. Wrap it in a
 * Function object to access it. */
struct FunctionContents;

/** A possibly-weak pointer to a Halide function. Take care to follow
 * the rules mentioned above. Preserves weakness/strength on copy.
 *
 * Note that Function objects are always strong pointers to Halide
 * functions.
 */
struct FunctionPtr {
    /** A strong and weak pointer to the group. Only one of these
     * should be non-zero. */
    // @{
    IntrusivePtr<FunctionGroup> strong;
    FunctionGroup *weak = nullptr;
    // @}

    /** The index of the function within the group. */
    int idx = 0;

    /** Get a pointer to the group this Function belongs to. */
    FunctionGroup *group() const {
        return weak ? weak : strong.get();
    }

    /** Get the opaque FunctionContents object this pointer refers
     * to. Wrap it in a Function to do anything interesting with it. */
    // @{
    FunctionContents *get() const;

    FunctionContents &operator*() const {
        return *get();
    }

    FunctionContents *operator->() const {
        return get();
    }
    // @}

    /** Convert from a strong reference to a weak reference. Does
     * nothing if the pointer is undefined, or if the reference is
     * already weak. */
    void weaken() {
        weak = group();
        strong = nullptr;
    }

    /** Convert from a weak reference to a strong reference. Does
     * nothing if the pointer is undefined, or if the reference is
     * already strong. */
    void strengthen() {
        strong = group();
        weak = nullptr;
    }

    /** Check if the reference is defined. */
    bool defined() const {
        return weak || strong.defined();
    }

    /** Check if two FunctionPtrs refer to the same Function. */
    bool same_as(const FunctionPtr &other) const {
        return idx == other.idx && group() == other.group();
    }

    /** Pointer comparison, for using FunctionPtrs as keys in maps and
     * sets. */
    bool operator<(const FunctionPtr &other) const {
        return get() < other.get();
    }
};

}
}

#endif
#ifndef HALIDE_PARAMETER_H
#define HALIDE_PARAMETER_H

/** \file
 * Defines the internal representation of parameters to halide piplines
 */

#ifndef HALIDE_BUFFER_H
#define HALIDE_BUFFER_H

/** \file
 * Defines a Buffer type that wraps from buffer_t and adds
 * functionality, and methods for more conveniently iterating over the
 * samples in a buffer_t outside of Halide code. */

#ifndef HALIDE_RUNTIME_BUFFER_H
#define HALIDE_RUNTIME_BUFFER_H

#include <memory>
#include <vector>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <string.h>


#ifdef _MSC_VER
#define HALIDE_ALLOCA _alloca
#else
#define HALIDE_ALLOCA __builtin_alloca
#endif

// gcc 5.1 has a false positive warning on this code
#if __GNUC__ == 5 && __GNUC_MINOR__ == 1
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

namespace Halide {
namespace Runtime {

// Forward-declare our Buffer class
template<typename T, int D> class Buffer;

// A helper to check if a parameter pack is entirely implicitly
// int-convertible to use with std::enable_if
template<typename ...Args>
struct AllInts : std::false_type {};

template<>
struct AllInts<> : std::true_type {};

template<typename T, typename ...Args>
struct AllInts<T, Args...> {
    static const bool value = std::is_convertible<T, int>::value && AllInts<Args...>::value;
};

// Floats and doubles are technically implicitly int-convertible, but
// doing so produces a warning we treat as an error, so just disallow
// it here.
template<typename ...Args>
struct AllInts<float, Args...> : std::false_type {};

template<typename ...Args>
struct AllInts<double, Args...> : std::false_type {};

/** A struct acting as a header for allocations owned by the Buffer
 * class itself. */
struct AllocationHeader {
    void (*deallocate_fn)(void *);
    std::atomic<int> ref_count {0};
};

/** This indicates how to deallocate the device for a Halide::Runtime::Buffer. */
enum struct BufferDeviceOwnership : int {
    Allocated,     ///> halide_device_free will be called when device ref count goes to zero
    WrappedNative, ///> halide_device_detach_native will be called when device ref count goes to zero
    Unmanaged,     ///> No free routine will be called when device ref count goes to zero
    AllocatedDeviceAndHost, ///> Call device_and_host_free when DeveRefCount goes to zero.
};

/** A similar struct for managing device allocations. */
struct DeviceRefCount {
    // This is only ever constructed when there's something to manage,
    // so start at one.
    std::atomic<int> count {1};
    BufferDeviceOwnership ownership{BufferDeviceOwnership::Allocated};
};

/** A templated Buffer class that wraps halide_buffer_t and adds
 * functionality. When using Halide from C++, this is the preferred
 * way to create input and output buffers. The overhead of using this
 * class relative to a naked halide_buffer_t is minimal - it uses another
 * ~16 bytes on the stack, and does no dynamic allocations when using
 * it to represent existing memory of a known maximum dimensionality.
 *
 * The template parameter T is the element type. For buffers where the
 * element type is unknown, or may vary, use void or const void.
 *
 * D is the maximum number of dimensions that can be represented using
 * space inside the class itself. Set it to the maximum dimensionality
 * you expect this buffer to be. If the actual dimensionality exceeds
 * this, heap storage is allocated to track the shape of the buffer. D
 * defaults to 4, which should cover nearly all usage.
 *
 * The class optionally allocates and owns memory for the image using
 * a shared pointer allocated with the provided allocator. If they are
 * null, malloc and free are used.  Any device-side allocation is
 * considered as owned if and only if the host-side allocation is
 * owned. */
template<typename T = void, int D = 4>
class Buffer {
    /** The underlying buffer_t */
    halide_buffer_t buf = {0};

    /** Some in-class storage for shape of the dimensions. */
    halide_dimension_t shape[D];

    /** The allocation owned by this Buffer. NULL if the Buffer does not
     * own the memory. */
    AllocationHeader *alloc = nullptr;

    /** A reference count for the device allocation owned by this
     * buffer. */
    mutable DeviceRefCount *dev_ref_count = nullptr;

    /** True if T is of type void or const void */
    static const bool T_is_void = std::is_same<typename std::remove_const<T>::type, void>::value;

    /** A type function that adds a const qualifier if T is a const type. */
    template<typename T2>
    using add_const_if_T_is_const = typename std::conditional<std::is_const<T>::value, const T2, T2>::type;

    /** T unless T is (const) void, in which case (const)
     * uint8_t. Useful for providing return types for operator() */
    using not_void_T = typename std::conditional<T_is_void,
                                                 add_const_if_T_is_const<uint8_t>,
                                                 T>::type;

    /** The type the elements are stored as. Equal to not_void_T
     * unless T is a pointer, in which case uint64_t. Halide stores
     * all pointer types as uint64s internally, even on 32-bit
     * systems. */
    using storage_T = typename std::conditional<std::is_pointer<T>::value, uint64_t, not_void_T>::type;

public:
    /** True if the Halide type is not void (or const void). */
    static constexpr bool has_static_halide_type = !T_is_void;

    /** Get the Halide type of T. Callers should not use the result if
     * has_static_halide_type is false. */
    static halide_type_t static_halide_type() {
        return halide_type_of<typename std::remove_cv<not_void_T>::type>();
    }

    /** Does this Buffer own the host memory it refers to? */
    bool owns_host_memory() const {
        return alloc != nullptr;
    }

private:
    /** Increment the reference count of any owned allocation */
    void incref() const {
        if (owns_host_memory()) {
            alloc->ref_count++;
        }
        if (buf.device) {
            if (!dev_ref_count) {
                // I seem to have a non-zero dev field but no
                // reference count for it. I must have been given a
                // device allocation by a Halide pipeline, and have
                // never been copied from since. Take sole ownership
                // of it.
                dev_ref_count = new DeviceRefCount;
            }
            dev_ref_count->count++;
        }
    }

    /** Decrement the reference count of any owned allocation and free host
     * and device memory if it hits zero. Sets alloc to nullptr. */
    void decref() {
        if (owns_host_memory()) {
            int new_count = --(alloc->ref_count);
            if (new_count == 0) {
                void (*fn)(void *) = alloc->deallocate_fn;
                fn(alloc);
            }
            buf.host = nullptr;
            alloc = nullptr;
            set_host_dirty(false);
        }
        decref_dev();
    }

    void decref_dev() {
        int new_count = 0;
        if (dev_ref_count) {
            new_count = --(dev_ref_count->count);
        }
        if (new_count == 0) {
            if (buf.device) {
                assert(!(alloc && device_dirty()) &&
                       "Implicitly freeing a dirty device allocation while a host allocation still lives. "
                       "Call device_free explicitly if you want to drop dirty device-side data. "
                       "Call copy_to_host explicitly if you want the data copied to the host allocation "
                       "before the device allocation is freed.");
                if (dev_ref_count && dev_ref_count->ownership == BufferDeviceOwnership::WrappedNative) {
                    buf.device_interface->detach_native(nullptr, &buf);
                } else if (dev_ref_count && dev_ref_count->ownership == BufferDeviceOwnership::AllocatedDeviceAndHost) {
                    buf.device_interface->device_and_host_free(nullptr, &buf);
                } else if (dev_ref_count == nullptr || dev_ref_count->ownership == BufferDeviceOwnership::Allocated) {
                    buf.device_interface->device_free(nullptr, &buf);
                }
            }
            if (dev_ref_count) {
                delete dev_ref_count;
            }
        }
        buf.device = 0;
        buf.device_interface = nullptr;
        dev_ref_count = nullptr;
    }

    void free_shape_storage() {
        if (buf.dim != shape) {
            delete[] buf.dim;
            buf.dim = nullptr;
        }
    }

    void make_shape_storage() {
        if (buf.dimensions <= D) {
            buf.dim = shape;
        } else {
            buf.dim = new halide_dimension_t[buf.dimensions];
        }
    }

    void copy_shape_from(const halide_buffer_t &other) {
        // All callers of this ensure that buf.dimensions == other.dimensions.
        make_shape_storage();
        for (int i = 0; i < buf.dimensions; i++) {
            buf.dim[i] = other.dim[i];
        }
    }

    template<typename T2, int D2>
    void move_shape_from(Buffer<T2, D2> &&other) {
        if (other.shape == other.buf.dim) {
            copy_shape_from(other.buf);
        } else {
            buf.dim = other.buf.dim;
            other.buf.dim = nullptr;
        }
    }

    /** Initialize the shape from a halide_buffer_t. */
    void initialize_from_buffer(const halide_buffer_t &b,
                                BufferDeviceOwnership ownership) {
        memcpy(&buf, &b, sizeof(halide_buffer_t));
        copy_shape_from(b);
        if (b.device) {
            dev_ref_count = new DeviceRefCount;
            dev_ref_count->ownership = ownership;
        }
    }

    /** Initialize the shape from a parameter pack of ints */
    template<typename ...Args>
    void initialize_shape(int next, int first, Args... rest) {
        buf.dim[next].min = 0;
        buf.dim[next].extent = first;
        if (next == 0) {
            buf.dim[next].stride = 1;
        } else {
            buf.dim[next].stride = buf.dim[next-1].stride * buf.dim[next-1].extent;
        }
        initialize_shape(next + 1, rest...);
    }

    /** Base case for the template recursion above. */
    void initialize_shape(int) {
    }

    /** Initialize the shape from a vector of extents */
    void initialize_shape(const std::vector<int> &sizes) {
        assert(sizes.size() <= std::numeric_limits<int>::max());
        int limit = (int)sizes.size();
        assert(limit <= dimensions());
        for (int i = 0; i < limit; i++) {
            buf.dim[i].min = 0;
            buf.dim[i].extent = sizes[i];
            if (i == 0) {
                buf.dim[i].stride = 1;
            } else {
                buf.dim[i].stride = buf.dim[i-1].stride * buf.dim[i-1].extent;
            }
        }
    }

    /** Initialize the shape from the static shape of an array */
    template<typename Array, size_t N>
    void initialize_shape_from_array_shape(int next, Array (&vals)[N]) {
        buf.dim[next].min = 0;
        buf.dim[next].extent = (int)N;
        if (next == 0) {
            buf.dim[next].stride = 1;
        } else {
            initialize_shape_from_array_shape(next - 1, vals[0]);
            buf.dim[next].stride = buf.dim[next - 1].stride * buf.dim[next - 1].extent;
        }
    }

    /** Base case for the template recursion above. */
    template<typename T2>
    void initialize_shape_from_array_shape(int, const T2 &) {
    }

    /** Get the dimensionality of a multi-dimensional C array */
    template<typename Array, size_t N>
    static int dimensionality_of_array(Array (&vals)[N]) {
        return dimensionality_of_array(vals[0]) + 1;
    }

    template<typename T2>
    static int dimensionality_of_array(const T2 &) {
        return 0;
    }

    /** Get the underlying halide_type_t of an array's element type. */
    template<typename Array, size_t N>
    static halide_type_t scalar_type_of_array(Array (&vals)[N]) {
        return scalar_type_of_array(vals[0]);
    }

    template<typename T2>
    static halide_type_t scalar_type_of_array(const T2 &) {
        return halide_type_of<typename std::remove_cv<T2>::type>();
    }

    /** Check if any args in a parameter pack are zero */
    template<typename ...Args>
    static bool any_zero(int first, Args... rest) {
        if (first == 0) return true;
        return any_zero(rest...);
    }

    static bool any_zero() {
        return false;
    }

    static bool any_zero(const std::vector<int> &v) {
        for (int i : v) {
            if (i == 0) return true;
        }
        return false;
    }

public:

    typedef T ElemType;

    /** Read-only access to the shape */
    class Dimension {
        const halide_dimension_t &d;
    public:
        /** The lowest coordinate in this dimension */
        HALIDE_ALWAYS_INLINE int min() const {
            return d.min;
        }

        /** The number of elements in memory you have to step over to
         * increment this coordinate by one. */
        HALIDE_ALWAYS_INLINE int stride() const {
            return d.stride;
        }

        /** The extent of the image along this dimension */
        HALIDE_ALWAYS_INLINE int extent() const {
            return d.extent;
        }

        /** The highest coordinate in this dimension */
        HALIDE_ALWAYS_INLINE int max() const {
            return min() + extent() - 1;
        }

        /** An iterator class, so that you can iterate over
         * coordinates in a dimensions using a range-based for loop. */
        struct iterator {
            int val;
            int operator*() const {return val;}
            bool operator!=(const iterator &other) const {return val != other.val;}
            iterator &operator++() {val++; return *this;}
        };

        /** An iterator that points to the min coordinate */
        HALIDE_ALWAYS_INLINE iterator begin() const {
            return {min()};
        }

        /** An iterator that points to one past the max coordinate */
        HALIDE_ALWAYS_INLINE iterator end() const {
            return {min() + extent()};
        }

        Dimension(const halide_dimension_t &dim) : d(dim) {};
    };

    /** Access the shape of the buffer */
    HALIDE_ALWAYS_INLINE Dimension dim(int i) const {
        return Dimension(buf.dim[i]);
    }

    /** Access to the mins, strides, extents. Will be deprecated. Do not use. */
    // @{
    int min(int i) const { return dim(i).min(); }
    int extent(int i) const { return dim(i).extent(); }
    int stride(int i) const { return dim(i).stride(); }
    // @}

    /** The total number of elements this buffer represents. Equal to
     * the product of the extents */
    size_t number_of_elements() const {
        size_t s = 1;
        for (int i = 0; i < dimensions(); i++) {
            s *= dim(i).extent();
        }
        return s;
    }

    /** Get the dimensionality of the buffer. */
    int dimensions() const {
        return buf.dimensions;
    }

    /** Get the type of the elements. */
    halide_type_t type() const {
        return buf.type;
    }

    /** A pointer to the element with the lowest address. If all
     * strides are positive, equal to the host pointer. */
    T *begin() const {
        ptrdiff_t index = 0;
        for (int i = 0; i < dimensions(); i++) {
            if (dim(i).stride() < 0) {
                index += (ptrdiff_t)dim(i).stride() * (ptrdiff_t)(dim(i).extent() - 1);
            }
        }
        return (T *)(buf.host + index * type().bytes());
    }

    /** A pointer to one beyond the element with the highest address. */
    T *end() const {
        ptrdiff_t index = 0;
        for (int i = 0; i < dimensions(); i++) {
            if (dim(i).stride() > 0) {
                index += (ptrdiff_t)dim(i).stride() * (ptrdiff_t)(dim(i).extent() - 1);
            }
        }
        index += 1;
        return (T *)(buf.host + index * type().bytes());
    }

    /** The total number of bytes spanned by the data in memory. */
    size_t size_in_bytes() const {
        return (size_t)((const uint8_t *)end() - (const uint8_t *)begin());
    }

    Buffer() {
        buf.type = static_halide_type();
        make_shape_storage();
    }

    /** Make a Buffer from a halide_buffer_t */
    Buffer(const halide_buffer_t &buf,
           BufferDeviceOwnership ownership = BufferDeviceOwnership::Unmanaged) {
        assert(T_is_void || buf.type == static_halide_type());
        initialize_from_buffer(buf, ownership);
    }

    /** Make a Buffer from a legacy buffer_t. */
    Buffer(const buffer_t &old_buf) {
        assert(!T_is_void && old_buf.elem_size == static_halide_type().bytes());
        buf.host = old_buf.host;
        buf.type = static_halide_type();
        int d;
        for (d = 0; d < 4 && old_buf.extent[d]; d++);
        buf.dimensions = d;
        make_shape_storage();
        for (int i = 0; i < d; i++) {
            buf.dim[i].min = old_buf.min[i];
            buf.dim[i].extent = old_buf.extent[i];
            buf.dim[i].stride = old_buf.stride[i];
        }
        buf.set_host_dirty(old_buf.host_dirty);
        assert(old_buf.dev == 0 && "Cannot construct a Halide::Runtime::Buffer from a legacy buffer_t with a device allocation. Use halide_upgrade_buffer_t to upgrade it to a halide_buffer_t first.");
    }

    /** Populate the fields of a legacy buffer_t using this
     * Buffer. Does not copy device metadata. */
    buffer_t make_legacy_buffer_t() const {
        buffer_t old_buf = {0};
        assert(!has_device_allocation() && "Cannot construct a legacy buffer_t from a Halide::Runtime::Buffer with a device allocation. Use halide_downgrade_buffer_t instead.");
        old_buf.host = buf.host;
        old_buf.elem_size = buf.type.bytes();
        assert(dimensions() <= 4 && "Cannot construct a legacy buffer_t from a Halide::Runtime::Buffer with more than four dimensions.");
        for (int i = 0; i < dimensions(); i++) {
            old_buf.min[i] = dim(i).min();
            old_buf.extent[i] = dim(i).extent();
            old_buf.stride[i] = dim(i).stride();
        }
        return old_buf;
    }

    /** Give Buffers access to the members of Buffers of different dimensionalities and types. */
    template<typename T2, int D2> friend class Buffer;

    /** Determine if if an Buffer<T, D> can be constructed from some other Buffer type.
     * If this can be determined at compile time, fail with a static assert; otherwise
     * return a boolean based on runtime typing. */
    template<typename T2, int D2>
    static bool can_convert_from(const Buffer<T2, D2> &other) {
        static_assert((!std::is_const<T2>::value || std::is_const<T>::value),
                      "Can't convert from a Buffer<const T> to a Buffer<T>");
        static_assert(std::is_same<typename std::remove_const<T>::type,
                                   typename std::remove_const<T2>::type>::value ||
                      T_is_void || Buffer<T2, D2>::T_is_void,
                      "type mismatch constructing Buffer");
        if (Buffer<T2, D2>::T_is_void && !T_is_void) {
            return other.type() == static_halide_type();
        }
        return true;
    }

    /** Fail an assertion at runtime or compile-time if an Buffer<T, D>
     * cannot be constructed from some other Buffer type. */
    template<typename T2, int D2>
    static void assert_can_convert_from(const Buffer<T2, D2> &other) {
        assert(can_convert_from(other));
    }

    /** Copy constructor. Does not copy underlying data. */
    Buffer(const Buffer<T, D> &other) : buf(other.buf),
                                        alloc(other.alloc) {
        other.incref();
        dev_ref_count = other.dev_ref_count;
        copy_shape_from(other.buf);
    }

    /** Construct a Buffer from a Buffer of different dimensionality
     * and type. Asserts that the type matches (at runtime, if one of
     * the types is void). Note that this constructor is
     * implicit. This, for example, lets you pass things like
     * Buffer<T> or Buffer<const void> to functions expected
     * Buffer<const T>. */
    template<typename T2, int D2>
    Buffer(const Buffer<T2, D2> &other) : buf(other.buf),
                                          alloc(other.alloc) {
        assert_can_convert_from(other);
        other.incref();
        dev_ref_count = other.dev_ref_count;
        copy_shape_from(other.buf);
    }

    /** Move constructor */
    Buffer(Buffer<T, D> &&other) : buf(other.buf),
                                   alloc(other.alloc),
                                   dev_ref_count(other.dev_ref_count) {
        other.dev_ref_count = nullptr;
        other.alloc = nullptr;
        other.buf.device = 0;
        other.buf.device_interface = nullptr;
        move_shape_from(std::forward<Buffer<T, D>>(other));
    }

    /** Move-construct a Buffer from a Buffer of different
     * dimensionality and type. Asserts that the types match (at
     * runtime if one of the types is void). */
    template<typename T2, int D2>
    Buffer(Buffer<T2, D2> &&other) : buf(other.buf),
                                     alloc(other.alloc),
                                     dev_ref_count(other.dev_ref_count) {
        other.dev_ref_count = nullptr;
        other.alloc = nullptr;
        other.buf.device = 0;
        other.buf.device_interface = nullptr;
        move_shape_from(std::forward<Buffer<T2, D2>>(other));
    }

    /** Assign from another Buffer of possibly-different
     * dimensionality and type. Asserts that the types match (at
     * runtime if one of the types is void). */
    template<typename T2, int D2>
    Buffer<T, D> &operator=(const Buffer<T2, D2> &other) {
        if ((const void *)this == (const void *)&other) {
            return *this;
        }
        assert_can_convert_from(other);
        other.incref();
        decref();
        dev_ref_count = other.dev_ref_count;
        alloc = other.alloc;
        free_shape_storage();
        buf = other.buf;
        copy_shape_from(other.buf);
        return *this;
    }

    /** Standard assignment operator */
    Buffer<T, D> &operator=(const Buffer<T, D> &other) {
        if (this == &other) {
            return *this;
        }
        other.incref();
        decref();
        dev_ref_count = other.dev_ref_count;
        alloc = other.alloc;
        free_shape_storage();
        buf = other.buf;
        copy_shape_from(other.buf);
        return *this;
    }

    /** Move from another Buffer of possibly-different
     * dimensionality and type. Asserts that the types match (at
     * runtime if one of the types is void). */
    template<typename T2, int D2>
    Buffer<T, D> &operator=(Buffer<T2, D2> &&other) {
        assert_can_convert_from(other);
        decref();
        alloc = other.alloc;
        other.alloc = nullptr;
        dev_ref_count = other.dev_ref_count;
        other.dev_ref_count = nullptr;
        free_shape_storage();
        buf = other.buf;
        other.buf.device = 0;
        other.buf.device_interface = nullptr;
        move_shape_from(std::forward<Buffer<T2, D2>>(other));
        return *this;
    }

    /** Standard move-assignment operator */
    Buffer<T, D> &operator=(Buffer<T, D> &&other) {
        decref();
        alloc = other.alloc;
        other.alloc = nullptr;
        dev_ref_count = other.dev_ref_count;
        other.dev_ref_count = nullptr;
        free_shape_storage();
        buf = other.buf;
        other.buf.device = 0;
        other.buf.device_interface = nullptr;
        move_shape_from(std::forward<Buffer<T, D>>(other));
        return *this;
    }

    /** Check the product of the extents fits in memory. */
    void check_overflow() {
        size_t size = type().bytes();
        for (int i = 0; i < dimensions(); i++) {
            size *= dim(i).extent();
        }
        // We allow 2^31 or 2^63 bytes, so drop the top bit.
        size = (size << 1) >> 1;
        for (int i = 0; i < dimensions(); i++) {
            size /= dim(i).extent();
        }
        assert(size == (size_t)type().bytes() && "Error: Overflow computing total size of buffer.");
    }

    /** Allocate memory for this Buffer. Drops the reference to any
     * owned memory. */
    void allocate(void *(*allocate_fn)(size_t) = nullptr,
                  void (*deallocate_fn)(void *) = nullptr) {
        if (!allocate_fn) {
            allocate_fn = malloc;
        }
        if (!deallocate_fn) {
            deallocate_fn = free;
        }

        // Drop any existing allocation
        deallocate();

        // Conservatively align images to 128 bytes. This is enough
        // alignment for all the platforms we might use.
        size_t size = size_in_bytes();
        const size_t alignment = 128;
        size = (size + alignment - 1) & ~(alignment - 1);
        alloc = (AllocationHeader *)allocate_fn(size + sizeof(AllocationHeader) + alignment - 1);
        alloc->deallocate_fn = deallocate_fn;
        alloc->ref_count = 1;
        uint8_t *unaligned_ptr = ((uint8_t *)alloc) + sizeof(AllocationHeader);
        buf.host = (uint8_t *)((uintptr_t)(unaligned_ptr + alignment - 1) & ~(alignment - 1));
    }

    /** Drop reference to any owned host or device memory, possibly
     * freeing it, if this buffer held the last reference to
     * it. Retains the shape of the buffer. Does nothing if this
     * buffer did not allocate its own memory. */
    void deallocate() {
        decref();
    }

    /** Drop reference to any owned device memory, possibly freeing it
     * if this buffer held the last reference to it. Asserts that
     * device_dirty is false. */
    void device_deallocate() {
        decref_dev();
    }

    /** Allocate a new image of the given size with a runtime
     * type. Only used when you do know what size you want but you
     * don't know statically what type the elements are. Pass zeroes
     * to make a buffer suitable for bounds query calls. */
    template<typename ...Args,
             typename = typename std::enable_if<AllInts<Args...>::value>::type>
    Buffer(halide_type_t t, int first, Args... rest) {
        if (!T_is_void) {
            assert(static_halide_type() == t);
        }
        buf.type = t;
        buf.dimensions = 1 + (int)(sizeof...(rest));
        make_shape_storage();
        initialize_shape(0, first, rest...);
        if (!any_zero(first, rest...)) {
            check_overflow();
            allocate();
        }
    }


    /** Allocate a new image of the given size. Pass zeroes to make a
     * buffer suitable for bounds query calls. */
    // @{

    // The overload with one argument is 'explicit', so that
    // (say) int is not implicitly convertable to Buffer<int>
    explicit Buffer(int first) {
        static_assert(!T_is_void,
                      "To construct an Buffer<void>, pass a halide_type_t as the first argument to the constructor");
        buf.type = static_halide_type();
        buf.dimensions = 1;
        make_shape_storage();
        initialize_shape(0, first);
        if (first != 0) {
            check_overflow();
            allocate();
        }
    }

    template<typename ...Args,
             typename = typename std::enable_if<AllInts<Args...>::value>::type>
    Buffer(int first, int second, Args... rest) {
        static_assert(!T_is_void,
                      "To construct an Buffer<void>, pass a halide_type_t as the first argument to the constructor");
        buf.type = static_halide_type();
        buf.dimensions = 2 + (int)(sizeof...(rest));
        make_shape_storage();
        initialize_shape(0, first, second, rest...);
        if (!any_zero(first, second, rest...)) {
            check_overflow();
            allocate();
        }
    }
    // @}

    /** Allocate a new image of unknown type using a vector of ints as the size. */
    Buffer(halide_type_t t, const std::vector<int> &sizes) {
        if (!T_is_void) {
            assert(static_halide_type() == t);
        }
        buf.type = t;
        buf.dimensions = (int)sizes.size();
        make_shape_storage();
        initialize_shape(sizes);
        if (!any_zero(sizes)) {
            check_overflow();
            allocate();
        }
    }

    /** Allocate a new image of known type using a vector of ints as the size. */
    Buffer(const std::vector<int> &sizes) {
        buf.type = static_halide_type();
        buf.dimensions = (int)sizes.size();
        make_shape_storage();
        initialize_shape(sizes);
        if (!any_zero(sizes)) {
            check_overflow();
            allocate();
        }
    }

    /** Make an Buffer that refers to a statically sized array. Does not
     * take ownership of the data, and does not set the host_dirty flag. */
    template<typename Array, size_t N>
    explicit Buffer(Array (&vals)[N]) {
        buf.dimensions = dimensionality_of_array(vals);
        buf.type = scalar_type_of_array(vals);
        buf.host = (uint8_t *)vals;
        make_shape_storage();
        initialize_shape_from_array_shape(buf.dimensions - 1, vals);
    }

    /** Initialize an Buffer of runtime type from a pointer and some
     * sizes. Assumes dense row-major packing and a min coordinate of
     * zero. Does not take ownership of the data and does not set the
     * host_dirty flag. */
    template<typename ...Args,
             typename = typename std::enable_if<AllInts<Args...>::value>::type>
    explicit Buffer(halide_type_t t, add_const_if_T_is_const<void> *data, int first, Args&&... rest) {
        if (!T_is_void) {
            assert(static_halide_type() == t);
        }
        buf.type = t;
        buf.dimensions = 1 + (int)(sizeof...(rest));
        buf.host = (uint8_t *)data;
        make_shape_storage();
        initialize_shape(0, first, int(rest)...);
    }

    /** Initialize an Buffer from a pointer and some sizes. Assumes
     * dense row-major packing and a min coordinate of zero. Does not
     * take ownership of the data and does not set the host_dirty flag. */
    template<typename ...Args,
             typename = typename std::enable_if<AllInts<Args...>::value>::type>
    explicit Buffer(T *data, int first, Args&&... rest) {
        buf.type = static_halide_type();
        buf.dimensions = 1 + (int)(sizeof...(rest));
        buf.host = (uint8_t *)data;
        make_shape_storage();
        initialize_shape(0, first, int(rest)...);
    }

    /** Initialize an Buffer from a pointer and a vector of
     * sizes. Assumes dense row-major packing and a min coordinate of
     * zero. Does not take ownership of the data and does not set the
     * host_dirty flag. */
    explicit Buffer(T *data, const std::vector<int> &sizes) {
        buf.type = static_halide_type();
        buf.dimensions = (int)sizes.size();
        buf.host = (uint8_t *)data;
        make_shape_storage();
        initialize_shape(sizes);
    }

    /** Initialize an Buffer of runtime type from a pointer and a
     * vector of sizes. Assumes dense row-major packing and a min
     * coordinate of zero. Does not take ownership of the data and
     * does not set the host_dirty flag. */
    explicit Buffer(halide_type_t t, add_const_if_T_is_const<void> *data, const std::vector<int> &sizes) {
        if (!T_is_void) {
            assert(static_halide_type() == t);
        }
        buf.type = t;
        buf.dimensions = (int)sizes.size();
        buf.host = (uint8_t *)data;
        make_shape_storage();
        initialize_shape(sizes);
    }

    /** Initialize an Buffer from a pointer to the min coordinate and
     * an array describing the shape.  Does not take ownership of the
     * data, and does not set the host_dirty flag. */
    explicit Buffer(halide_type_t t, add_const_if_T_is_const<void> *data, int d, const halide_dimension_t *shape) {
        if (!T_is_void) {
            assert(static_halide_type() == t);
        }
        buf.type = t;
        buf.dimensions = d;
        buf.host = (uint8_t *)data;
        make_shape_storage();
        for (int i = 0; i < d; i++) {
            buf.dim[i] = shape[i];
        }
    }

    /** Initialize an Buffer from a pointer to the min coordinate and
     * an array describing the shape.  Does not take ownership of the
     * data and does not set the host_dirty flag. */
    explicit Buffer(T *data, int d, const halide_dimension_t *shape) {
        buf.type = halide_type_of<typename std::remove_cv<T>::type>();
        buf.dimensions = d;
        buf.host = (uint8_t *)data;
        make_shape_storage();
        for (int i = 0; i < d; i++) {
            buf.dim[i] = shape[i];
        }
    }

    /** Destructor. Will release any underlying owned allocation if
     * this is the last reference to it. Will assert fail if there are
     * weak references to this Buffer outstanding. */
    ~Buffer() {
        free_shape_storage();
        decref();
    }

    /** Get a pointer to the raw buffer_t this wraps. */
    // @{
    halide_buffer_t *raw_buffer() {
        return &buf;
    }

    const halide_buffer_t *raw_buffer() const {
        return &buf;
    }
    // @}

    /** Provide a cast operator to halide_buffer_t *, so that
     * instances can be passed directly to Halide filters. */
    operator halide_buffer_t *() {
        return &buf;
    }

    /** Return a typed reference to this Buffer. Useful for converting
     * a reference to a Buffer<void> to a reference to, for example, a
     * Buffer<const uint8_t>. Does a runtime assert if the source
     * buffer type is void. */
    template<typename T2, int D2 = D,
             typename = typename std::enable_if<(D2 <= D)>::type>
    Buffer<T2, D2> &as() & {
        Buffer<T2, D>::assert_can_convert_from(*this);
        return *((Buffer<T2, D2> *)this);
    }

    /** Return a const typed reference to this Buffer. Useful for
     * converting a conference reference to one Buffer type to a const
     * reference to another Buffer type. Does a runtime assert if the
     * source buffer type is void. */
    template<typename T2, int D2 = D,
             typename = typename std::enable_if<(D2 <= D)>::type>
    const Buffer<T2, D2> &as() const &  {
        Buffer<T2, D>::assert_can_convert_from(*this);
        return *((const Buffer<T2, D2> *)this);
    }

    /** Returns this rval Buffer with a different type attached. Does
     * a dynamic type check if the source type is void. */
    template<typename T2, int D2 = D>
    Buffer<T2, D2> as() && {
        Buffer<T2, D2>::assert_can_convert_from(*this);
        return *((Buffer<T2, D2> *)this);
    }

    /** Conventional names for the first three dimensions. */
    // @{
    int width() const {
        return (dimensions() > 0) ? dim(0).extent() : 1;
    }
    int height() const {
        return (dimensions() > 1) ? dim(1).extent() : 1;
    }
    int channels() const {
        return (dimensions() > 2) ? dim(2).extent() : 1;
    }
    // @}

    /** Conventional names for the min and max value of each dimension */
    // @{
    int left() const {
        return dim(0).min();
    }

    int right() const {
        return dim(0).max();
    }

    int top() const {
        return dim(1).min();
    }

    int bottom() const {
        return dim(1).max();
    }
    // @}

    /** Make a new image which is a deep copy of this image. Use crop
     * or slice followed by copy to make a copy of only a portion of
     * the image. The new image uses the same memory layout as the
     * original, with holes compacted away. */
    Buffer<T, D> copy(void *(*allocate_fn)(size_t) = nullptr,
                      void (*deallocate_fn)(void *) = nullptr) const {
        Buffer<T, D> dst = make_with_shape_of(*this, allocate_fn, deallocate_fn);
        dst.copy_from(*this);
        return dst;
    }

    /** Fill a Buffer with the values at the same coordinates in
     * another Buffer. Restricts itself to coordinates contained
     * within the intersection of the two buffers. If the two Buffers
     * are not in the same coordinate system, you will need to
     * translate the argument Buffer first. E.g. if you're blitting a
     * sprite onto a framebuffer, you'll want to translate the sprite
     * to the correct location first like so: \code
     * framebuffer.copy_from(sprite.translated({x, y})); \endcode
    */
    template<typename T2, int D2>
    void copy_from(const Buffer<T2, D2> &other) {
        assert(!device_dirty() && "Cannot call Halide::Runtime::Buffer::copy_from on a device dirty destination.");
        assert(!other.device_dirty() && "Cannot call Halide::Runtime::Buffer::copy_from on a device dirty source.");

        Buffer<const T, D> src(other);
        Buffer<T, D> dst(*this);

        assert(src.dimensions() == dst.dimensions());

        // Trim the copy to the region in common
        for (int i = 0; i < dimensions(); i++) {
            int min_coord = std::max(dst.dim(i).min(), src.dim(i).min());
            int max_coord = std::min(dst.dim(i).max(), src.dim(i).max());
            if (max_coord < min_coord) {
                // The buffers do not overlap.
                return;
            }
            dst.crop(i, min_coord, max_coord - min_coord + 1);
            src.crop(i, min_coord, max_coord - min_coord + 1);
        }

        // If T is void, we need to do runtime dispatch to an
        // appropriately-typed lambda. We're copying, so we only care
        // about the element size.
        if (type().bytes() == 1) {
            using MemType = uint8_t;
            auto &typed_dst = (Buffer<MemType, D> &)dst;
            auto &typed_src = (Buffer<const MemType, D> &)src;
            typed_dst.for_each_value([&](MemType &dst, MemType src) {dst = src;}, typed_src);
        } else if (type().bytes() == 2) {
            using MemType = uint16_t;
            auto &typed_dst = (Buffer<MemType, D> &)dst;
            auto &typed_src = (Buffer<const MemType, D> &)src;
            typed_dst.for_each_value([&](MemType &dst, MemType src) {dst = src;}, typed_src);
        } else if (type().bytes() == 4) {
            using MemType = uint32_t;
            auto &typed_dst = (Buffer<MemType, D> &)dst;
            auto &typed_src = (Buffer<const MemType, D> &)src;
            typed_dst.for_each_value([&](MemType &dst, MemType src) {dst = src;}, typed_src);
        } else if (type().bytes() == 8) {
            using MemType = uint64_t;
            auto &typed_dst = (Buffer<MemType, D> &)dst;
            auto &typed_src = (Buffer<const MemType, D> &)src;
            typed_dst.for_each_value([&](MemType &dst, MemType src) {dst = src;}, typed_src);
        } else {
            assert(false && "type().bytes() must be 1, 2, 4, or 8");
        }
        set_host_dirty();
    }

    /** Make an image that refers to a sub-range of this image along
     * the given dimension. Does not assert the crop region is within
     * the existing bounds. The cropped image drops any device
     * handle. */
    Buffer<T, D> cropped(int d, int min, int extent) const {
        // Make a fresh copy of the underlying buffer (but not a fresh
        // copy of the allocation, if there is one).
        Buffer<T, D> im = *this;
        im.crop(d, min, extent);
        return im;
    }

    /** Crop an image in-place along the given dimension. */
    void crop(int d, int min, int extent) {
        // assert(dim(d).min() <= min);
        // assert(dim(d).max() >= min + extent - 1);
        int shift = min - dim(d).min();
        if (shift) {
            device_deallocate();
        }
        if (buf.host != nullptr) {
            buf.host += shift * dim(d).stride() * type().bytes();
        }
        buf.dim[d].min = min;
        buf.dim[d].extent = extent;
    }

    /** Make an image that refers to a sub-rectangle of this image along
     * the first N dimensions. Does not assert the crop region is within
     * the existing bounds. The cropped image drops any device handle. */
    Buffer<T, D> cropped(const std::vector<std::pair<int, int>> &rect) const {
        // Make a fresh copy of the underlying buffer (but not a fresh
        // copy of the allocation, if there is one).
        Buffer<T, D> im = *this;
        im.crop(rect);
        return im;
    }

    /** Crop an image in-place along the first N dimensions. */
    void crop(const std::vector<std::pair<int, int>> &rect) {
        assert(rect.size() <= std::numeric_limits<int>::max());
        int limit = (int)rect.size();
        assert(limit <= dimensions());
        for (int i = 0; i < limit; i++) {
            crop(i, rect[i].first, rect[i].second);
        }
    }

    /** Make an image which refers to the same data with using
     * translated coordinates in the given dimension. Positive values
     * move the image data to the right or down relative to the
     * coordinate system. Drops any device handle. */
    Buffer<T, D> translated(int d, int dx) const {
        Buffer<T, D> im = *this;
        im.translate(d, dx);
        return im;
    }

    /** Translate an image in-place along one dimension */
    void translate(int d, int delta) {
        device_deallocate();
        buf.dim[d].min += delta;
    }

    /** Make an image which refers to the same data translated along
     * the first N dimensions. */
    Buffer<T, D> translated(const std::vector<int> &delta) {
        Buffer<T, D> im = *this;
        im.translate(delta);
        return im;
    }

    /** Translate an image along the first N dimensions */
    void translate(const std::vector<int> &delta) {
        device_deallocate();
        assert(delta.size() <= std::numeric_limits<int>::max());
        int limit = (int)delta.size();
        assert(limit <= dimensions());
        for (int i = 0; i < limit; i++) {
            translate(i, delta[i]);
        }
    }

    /** Set the min coordinate of an image in the first N dimensions */
    template<typename ...Args>
    void set_min(Args... args) {
        assert(sizeof...(args) <= (size_t)dimensions());
        device_deallocate();
        const int x[] = {args...};
        for (size_t i = 0; i < sizeof...(args); i++) {
            buf.dim[i].min = x[i];
        }
    }

    /** Test if a given coordinate is within the the bounds of an image */
    template<typename ...Args>
    bool contains(Args... args) {
        assert(sizeof...(args) <= (size_t)dimensions());
        const int x[] = {args...};
        for (size_t i = 0; i < sizeof...(args); i++) {
            if (x[i] < dim(i).min() || x[i] > dim(i).max()) {
                return false;
            }
        }
        return true;
    }

    /** Make an image which refers to the same data using a different
     * ordering of the dimensions. */
    Buffer<T, D> transposed(int d1, int d2) const {
        Buffer<T, D> im = *this;
        im.transpose(d1, d2);
        return im;
    }

    /** Transpose an image in-place */
    void transpose(int d1, int d2) {
        std::swap(buf.dim[d1], buf.dim[d2]);
    }

    /** Make a lower-dimensional image that refers to one slice of this
     * image. */
    Buffer<T, D> sliced(int d, int pos) const {
        Buffer<T, D> im = *this;
        im.slice(d, pos);
        return im;
    }

    /** Slice an image in-place */
    void slice(int d, int pos) {
        // assert(pos >= dim(d).min() && pos <= dim(d).max());
        device_deallocate();
        buf.dimensions--;
        int shift = pos - dim(d).min();
        assert(buf.device == 0 || shift == 0);
        if (buf.host != nullptr) {
            buf.host += shift * dim(d).stride() * type().bytes();
        }
        for (int i = d; i < dimensions(); i++) {
            buf.dim[i] = buf.dim[i+1];
        }
        buf.dim[buf.dimensions] = {0, 0, 0};
    }

    /** Make a new image that views this image as a single slice in a
     * higher-dimensional space. The new dimension has extent one and
     * the given min. This operation is the opposite of slice. As an
     * example, the following condition is true:
     *
     \code
     im2 = im.embedded(1, 17);
     &im(x, y, c) == &im2(x, 17, y, c);
     \endcode
     */
    Buffer<T, D> embedded(int d, int pos) const {
        assert(d >= 0 && d <= dimensions());
        Buffer<T, D> im(*this);
        im.embed(d, pos);
        return im;
    }

    /** Embed an image in-place, increasing the
     * dimensionality. */
    void embed(int d, int pos) {
        assert(d >= 0 && d <= dimensions());
        add_dimension();
        translate(dimensions() - 1, pos);
        for (int i = dimensions() - 1; i > d; i--) {
            transpose(i, i-1);
        }
    }

    /** Add a new dimension with a min of zero and an extent of
     * one. The stride is the extent of the outermost dimension times
     * its stride. The new dimension is the last dimension. This is a
     * special case of embed. */
    void add_dimension() {
        const int dims = buf.dimensions;
        buf.dimensions++;
        if (buf.dim != shape) {
            // We're already on the heap. Reallocate.
            halide_dimension_t *new_shape = new halide_dimension_t[buf.dimensions];
            for (int i = 0; i < dims; i++) {
                new_shape[i] = buf.dim[i];
            }
            delete[] buf.dim;
            buf.dim = new_shape;
        } else if (dims == D) {
            // Transition from the in-class storage to the heap
            make_shape_storage();
            for (int i = 0; i < dims; i++) {
                buf.dim[i] = shape[i];
            }
        } else {
            // We still fit in the class
        }
        buf.dim[dims] = {0, 1, 0};
        if (dims == 0) {
            buf.dim[dims].stride = 1;
        } else {
            buf.dim[dims].stride = buf.dim[dims-1].extent * buf.dim[dims-1].stride;
        }
    }

    /** Add a new dimension with a min of zero, an extent of one, and
     * the specified stride. The new dimension is the last
     * dimension. This is a special case of embed. */
    void add_dimension_with_stride(int s) {
        add_dimension();
        buf.dim[buf.dimensions-1].stride = s;
    }

    /** Methods for managing any GPU allocation. */
    // @{
    void set_host_dirty(bool v = true) {
        assert((!v || !device_dirty()) && "Cannot set host dirty when device is already dirty.");
        buf.set_host_dirty(v);
    }

    bool device_dirty() const {
        return buf.device_dirty();
    }

    bool host_dirty() const {
        return buf.host_dirty();
    }

    void set_device_dirty(bool v = true) {
        assert((!v || !host_dirty()) && "Cannot set device dirty when host is already dirty.");
        buf.set_device_dirty(v);
    }

    int copy_to_host(void *ctx = nullptr) {
        if (device_dirty()) {
            return buf.device_interface->copy_to_host(ctx, &buf);
        }
        return 0;
    }

    int copy_to_device(const struct halide_device_interface_t *device_interface, void *ctx = nullptr) {
        if (host_dirty()) {
            return device_interface->copy_to_device(ctx, &buf, device_interface);
        }
        return 0;
    }

    int device_malloc(const struct halide_device_interface_t *device_interface, void *ctx = nullptr) {
        return device_interface->device_malloc(ctx, &buf, device_interface);
    }

    int device_free(void *ctx = nullptr) {
        if (dev_ref_count) {
            assert(dev_ref_count->ownership == BufferDeviceOwnership::Allocated &&
                   "Can't call device_free on an unmanaged or wrapped native device handle. "
                   "Free the source allocation or call device_detach_native instead.");
            // Multiple people may be holding onto this dev field
            assert(dev_ref_count->count == 1 &&
                   "Multiple Halide::Runtime::Buffer objects share this device "
                   "allocation. Freeing it would create dangling references. "
                   "Don't call device_free on Halide buffers that you have copied or "
                   "passed by value.");
        }
        int ret = 0;
        if (buf.device_interface) {
            ret = buf.device_interface->device_free(ctx, &buf);
        }
        if (dev_ref_count) {
            delete dev_ref_count;
            dev_ref_count = nullptr;
        }
        return ret;
    }

    int device_wrap_native(const struct halide_device_interface_t *device_interface,
                           uint64_t handle, void *ctx = nullptr) {
        assert(device_interface);
        dev_ref_count = new DeviceRefCount;
        dev_ref_count->ownership = BufferDeviceOwnership::WrappedNative;
        return device_interface->wrap_native(ctx, &buf, handle, device_interface);
    }

    int device_detach_native(void *ctx = nullptr) {
        assert(dev_ref_count &&
               dev_ref_count->ownership == BufferDeviceOwnership::WrappedNative &&
               "Only call device_detach_native on buffers wrapping a native "
               "device handle via device_wrap_native. This buffer was allocated "
               "using device_malloc, or is unmanaged. "
               "Call device_free or free the original allocation instead.");
        // Multiple people may be holding onto this dev field
        assert(dev_ref_count->count == 1 &&
               "Multiple Halide::Runtime::Buffer objects share this device "
               "allocation. Freeing it could create dangling references. "
               "Don't call device_detach_native on Halide buffers that you "
               "have copied or passed by value.");
        int ret = 0;
        if (buf.device_interface) {
            ret = buf.device_interface->detach_native(ctx, &buf);
        }
        delete dev_ref_count;
        dev_ref_count = nullptr;
        return ret;
    }

    int device_and_host_malloc(const struct halide_device_interface_t *device_interface, void *ctx = nullptr) {
        return device_interface->device_and_host_malloc(ctx, &buf, device_interface);
    }

    int device_and_host_free(const struct halide_device_interface_t *device_interface, void *ctx = nullptr) {
        if (dev_ref_count) {
            assert(dev_ref_count->ownership == BufferDeviceOwnership::AllocatedDeviceAndHost &&
                   "Can't call device_and_host_free on a device handle not allocated with device_and_host_malloc. "
                   "Free the source allocation or call device_detach_native instead.");
            // Multiple people may be holding onto this dev field
            assert(dev_ref_count->count == 1 &&
                   "Multiple Halide::Runtime::Buffer objects share this device "
                   "allocation. Freeing it would create dangling references. "
                   "Don't call device_and_host_free on Halide buffers that you have copied or "
                   "passed by value.");
        }
        int ret = 0;
        if (buf.device_interface) {
            ret = buf.device_interface->device_and_host_free(ctx, &buf);
        }
        if (dev_ref_count) {
            delete dev_ref_count;
            dev_ref_count = nullptr;
        }
        return ret;
    }

    int device_sync(void *ctx = nullptr) {
        if (buf.device_interface) {
            return buf.device_interface->device_sync(ctx, &buf);
        } else {
            return 0;
        }
    }

    bool has_device_allocation() const {
        return buf.device != 0;
    }

    /** Return the method by which the device field is managed. */
    BufferDeviceOwnership device_ownership() const {
        if (dev_ref_count == nullptr) {
            return BufferDeviceOwnership::Allocated;
        }
        return dev_ref_count->ownership;
    }
    // @}

    /** If you use the (x, y, c) indexing convention, then Halide
     * Buffers are stored planar by default. This function constructs
     * an interleaved RGB or RGBA image that can still be indexed
     * using (x, y, c). Passing it to a generator requires that the
     * generator has been compiled with support for interleaved (also
     * known as packed or chunky) memory layouts. */
    static Buffer<void, D> make_interleaved(halide_type_t t, int width, int height, int channels) {
        Buffer<void, D> im(t, channels, width, height);
        im.transpose(0, 1);
        im.transpose(1, 2);
        return im;
    }

    /** If you use the (x, y, c) indexing convention, then Halide
     * Buffers are stored planar by default. This function constructs
     * an interleaved RGB or RGBA image that can still be indexed
     * using (x, y, c). Passing it to a generator requires that the
     * generator has been compiled with support for interleaved (also
     * known as packed or chunky) memory layouts. */
    static Buffer<T, D> make_interleaved(int width, int height, int channels) {
        Buffer<T, D> im(channels, width, height);
        im.transpose(0, 1);
        im.transpose(1, 2);
        return im;
    }

    /** Wrap an existing interleaved image. */
    static Buffer<add_const_if_T_is_const<void>, D>
    make_interleaved(halide_type_t t, T *data, int width, int height, int channels) {
        Buffer<add_const_if_T_is_const<void>, D> im(t, data, channels, width, height);
        im.transpose(0, 1);
        im.transpose(1, 2);
        return im;
    }

    /** Wrap an existing interleaved image. */
    static Buffer<T, D> make_interleaved(T *data, int width, int height, int channels) {
        Buffer<T, D> im(data, channels, width, height);
        im.transpose(0, 1);
        im.transpose(1, 2);
        return im;
    }

    /** Make a zero-dimensional Buffer */
    static Buffer<add_const_if_T_is_const<void>, D> make_scalar(halide_type_t t) {
        Buffer<add_const_if_T_is_const<void>, 1> buf(t, 1);
        buf.slice(0, 0);
        return buf;
    }

    /** Make a zero-dimensional Buffer */
    static Buffer<T, D> make_scalar() {
        Buffer<T, 1> buf(1);
        buf.slice(0, 0);
        return buf;
    }

    /** Make a buffer with the same shape and memory nesting order as
     * another buffer. It may have a different type. */
    template<typename T2, int D2>
    static Buffer<T, D> make_with_shape_of(Buffer<T2, D2> src,
                                           void *(*allocate_fn)(size_t) = nullptr,
                                           void (*deallocate_fn)(void *) = nullptr) {
        // Reorder the dimensions of src to have strides in increasing order
        std::vector<int> swaps;
        for (int i = src.dimensions()-1; i > 0; i--) {
            for (int j = i; j > 0; j--) {
                if (src.dim(j-1).stride() > src.dim(j).stride()) {
                    src.transpose(j-1, j);
                    swaps.push_back(j);
                }
            }
        }

        // Rewrite the strides to be dense (this messes up src, which
        // is why we took it by value).
        halide_dimension_t *shape = src.buf.dim;
        for (int i = 0; i < src.dimensions(); i++) {
            if (i == 0) {
                shape[i].stride = 1;
            } else {
                shape[i].stride = shape[i-1].extent * shape[i-1].stride;
            }
        }

        // Undo the dimension reordering
        while (!swaps.empty()) {
            int j = swaps.back();
            std::swap(shape[j-1], shape[j]);
            swaps.pop_back();
        }

        Buffer<T, D> dst(nullptr, src.dimensions(), shape);
        dst.allocate(allocate_fn, deallocate_fn);

        return dst;
    }

private:

    template<typename ...Args>
    HALIDE_ALWAYS_INLINE
    ptrdiff_t offset_of(int d, int first, Args... rest) const {
        return offset_of(d+1, rest...) + this->buf.dim[d].stride * (first - this->buf.dim[d].min);
    }

    HALIDE_ALWAYS_INLINE
    ptrdiff_t offset_of(int d) const {
        return 0;
    }

    template<typename ...Args>
    HALIDE_ALWAYS_INLINE
    storage_T *address_of(Args... args) const {
        if (T_is_void) {
            return (storage_T *)(this->buf.host) + offset_of(0, args...) * type().bytes();
        } else {
            return (storage_T *)(this->buf.host) + offset_of(0, args...);
        }
    }

    HALIDE_ALWAYS_INLINE
    ptrdiff_t offset_of(const int *pos) const {
        ptrdiff_t offset = 0;
        for (int i = this->dimensions() - 1; i >= 0; i--) {
            offset += this->buf.dim[i].stride * (pos[i] - this->buf.dim[i].min);
        }
        return offset;
    }

    HALIDE_ALWAYS_INLINE
    storage_T *address_of(const int *pos) const {
        if (T_is_void) {
            return (storage_T *)this->buf.host + offset_of(pos) * type().bytes();
        } else {
            return (storage_T *)this->buf.host + offset_of(pos);
        }
    }

public:

    /** Get a pointer to the address of the min coordinate. */
    // @{
    T *data() {
        return (T *)(this->buf.host);
    }

    const T *data() const {
        return (const T *)(this->buf.host);
    }
    // @}

    /** Access elements. Use im(...) to get a reference to an element,
     * and use &im(...) to get the address of an element. If you pass
     * fewer arguments than the buffer has dimensions, the rest are
     * treated as their min coordinate. The non-const versions set the
     * host_dirty flag to true.
     */
    //@{
    template<typename ...Args,
             typename = typename std::enable_if<AllInts<Args...>::value>::type>
    HALIDE_ALWAYS_INLINE
    const not_void_T &operator()(int first, Args... rest) const {
        static_assert(!T_is_void,
                      "Cannot use operator() on Buffer<void> types");
        assert(!device_dirty());
        return *((const not_void_T *)(address_of(first, rest...)));
    }

    HALIDE_ALWAYS_INLINE
    const not_void_T &
    operator()() const {
        static_assert(!T_is_void,
                      "Cannot use operator() on Buffer<void> types");
        assert(!device_dirty());
        return *((const not_void_T *)(data()));
    }

    HALIDE_ALWAYS_INLINE
    const not_void_T &
    operator()(const int *pos) const {
        static_assert(!T_is_void,
                      "Cannot use operator() on Buffer<void> types");
        assert(!device_dirty());
        return *((const not_void_T *)(address_of(pos)));
    }

    template<typename ...Args,
             typename = typename std::enable_if<AllInts<Args...>::value>::type>
    HALIDE_ALWAYS_INLINE
    not_void_T &operator()(int first, Args... rest) {
        static_assert(!T_is_void,
                      "Cannot use operator() on Buffer<void> types");
        set_host_dirty();
        return *((not_void_T *)(address_of(first, rest...)));
    }

    HALIDE_ALWAYS_INLINE
    not_void_T &
    operator()() {
        static_assert(!T_is_void,
                      "Cannot use operator() on Buffer<void> types");
        set_host_dirty();
        return *((not_void_T *)(data()));
    }

    HALIDE_ALWAYS_INLINE
    not_void_T &
    operator()(const int *pos) {
        static_assert(!T_is_void,
                      "Cannot use operator() on Buffer<void> types");
        set_host_dirty();
        return *((not_void_T *)(address_of(pos)));
    }
    // @}

    void fill(not_void_T val) {
        set_host_dirty();
        for_each_value([=](T &v) {v = val;});
    }

private:
    /** Helper functions for for_each_value. */
    // @{
    template<int N>
    struct for_each_value_task_dim {
        int extent;
        int stride[N];
    };

    // Given an array of strides, and a bunch of pointers to pointers
    // (all of different types), advance the pointers using the
    // strides.
    template<typename Ptr, typename ...Ptrs>
    static void advance_ptrs(const int *stride, Ptr *ptr, Ptrs... ptrs) {
        (*ptr) += *stride;
        advance_ptrs(stride + 1, ptrs...);
    }

    static void advance_ptrs(const int *) {}

    // Same as the above, but just increments the pointers.
    template<typename Ptr, typename ...Ptrs>
    static void increment_ptrs(Ptr *ptr, Ptrs... ptrs) {
        (*ptr)++;
        increment_ptrs(ptrs...);
    }

    static void increment_ptrs() {}

    // Given a bunch of pointers to buffers of different types, read
    // out their strides in the d'th dimension, and assert that their
    // sizes match in that dimension.
    template<typename T2, int D2, typename ...Args>
    void extract_strides(int d, int *strides, const Buffer<T2, D2> *first, Args... rest) {
        assert(first->dimensions() == dimensions());
        assert(first->dim(d).min() == dim(d).min() &&
               first->dim(d).max() == dim(d).max());
        *strides++ = first->dim(d).stride();
        extract_strides(d, strides, rest...);
    }

    void extract_strides(int d, int *strides) {}

    // The template function that constructs the loop nest for for_each_value
    template<int d, bool innermost_strides_are_one, typename Fn, typename... Ptrs>
    static void for_each_value_helper(Fn &&f, const for_each_value_task_dim<sizeof...(Ptrs)> *t, Ptrs... ptrs) {
        if (d == -1) {
            f((*ptrs)...);
        } else {
            for (int i = t[d].extent; i != 0; i--) {
                for_each_value_helper<(d >= 0 ? d - 1 : -1), innermost_strides_are_one>(f, t, ptrs...);
                if (d == 0 && innermost_strides_are_one) {
                    // It helps with auto-vectorization to statically
                    // know the addresses are one apart in memory.
                    increment_ptrs((&ptrs)...);
                } else {
                    advance_ptrs(t[d].stride, (&ptrs)...);
                }
            }
        }
    }

    template<bool innermost_strides_are_one, typename Fn, typename... Ptrs>
    static void for_each_value_helper(Fn &&f, int d, const for_each_value_task_dim<sizeof...(Ptrs)> *t, Ptrs... ptrs) {
        // When we hit a low dimensionality, switch from runtime
        // recursion to template recursion.
        if (d == -1) {
            for_each_value_helper<-1, innermost_strides_are_one>(f, t, ptrs...);
        } else if (d == 0) {
            for_each_value_helper<0, innermost_strides_are_one>(f, t, ptrs...);
        } else if (d == 1) {
            for_each_value_helper<1, innermost_strides_are_one>(f, t, ptrs...);
        } else if (d == 2) {
            for_each_value_helper<2, innermost_strides_are_one>(f, t, ptrs...);
        } else {
            for (int i = t[d].extent; i != 0; i--) {
                for_each_value_helper<innermost_strides_are_one>(f, d-1, t, ptrs...);
                advance_ptrs(t[d].stride, (&ptrs)...);
            }
        }
    }
    // @}

public:
    /** Call a function on every value in the buffer, and the
     * corresponding values in some number of other buffers of the
     * same size. The function should take a reference, const
     * reference, or value of the correct type for each buffer. This
     * effectively lifts a function of scalars to an element-wise
     * function of buffers. This produces code that the compiler can
     * autovectorize. This is slightly cheaper than for_each_element,
     * because it does not need to track the coordinates. */
    template<typename Fn, typename ...Args, int N = sizeof...(Args) + 1>
    void for_each_value(Fn &&f, Args... other_buffers) {
        for_each_value_task_dim<N> *t =
            (for_each_value_task_dim<N> *)HALIDE_ALLOCA((dimensions()+1) * sizeof(for_each_value_task_dim<N>));
        for (int i = 0; i <= dimensions(); i++) {
            for (int j = 0; j < N; j++) {
                t[i].stride[j] = 0;
            }
            t[i].extent = 1;
        }

        for (int i = 0; i < dimensions(); i++) {
            extract_strides(i, t[i].stride, this, &other_buffers...);
            t[i].extent = dim(i).extent();
            // Order the dimensions by stride, so that the traversal is cache-coherent.
            for (int j = i; j > 0 && t[j].stride[0] < t[j-1].stride[0]; j--) {
                std::swap(t[j], t[j-1]);
            }
        }

        // flatten dimensions where possible to make a larger inner
        // loop for autovectorization.
        int d = dimensions();
        for (int i = 1; i < d; i++) {
            bool flat = true;
            for (int j = 0; j < N; j++) {
                flat = flat && t[i-1].stride[j] * t[i-1].extent == t[i].stride[j];
            }
            if (flat) {
                t[i-1].extent *= t[i].extent;
                for (int j = i; j < dimensions(); j++) {
                    t[j] = t[j+1];
                }
                i--;
                d--;
            }
        }

        bool innermost_strides_are_one = false;
        if (dimensions() > 0) {
            innermost_strides_are_one = true;
            for (int j = 0; j < N; j++) {
                innermost_strides_are_one &= t[0].stride[j] == 1;
            }
        }

        if (innermost_strides_are_one) {
            for_each_value_helper<true>(f, dimensions() - 1, t, begin(), (other_buffers.begin())...);
        } else {
            for_each_value_helper<false>(f, dimensions() - 1, t, begin(), (other_buffers.begin())...);
        }
    }

private:

    // Helper functions for for_each_element
    struct for_each_element_task_dim {
        int min, max;
    };

    /** If f is callable with this many args, call it. The first
     * argument is just to make the overloads distinct. Actual
     * overload selection is done using the enable_if. */
    template<typename Fn,
             typename ...Args,
             typename = decltype(std::declval<Fn>()(std::declval<Args>()...))>
    HALIDE_ALWAYS_INLINE
    static void for_each_element_variadic(int, int, const for_each_element_task_dim *, Fn &&f, Args... args) {
        f(args...);
    }

    /** If the above overload is impossible, we add an outer loop over
     * an additional argument and try again. */
    template<typename Fn,
             typename ...Args>
    HALIDE_ALWAYS_INLINE
    static void for_each_element_variadic(double, int d, const for_each_element_task_dim *t, Fn &&f, Args... args) {
        for (int i = t[d].min; i <= t[d].max; i++) {
            for_each_element_variadic(0, d - 1, t, std::forward<Fn>(f), i, args...);
        }
    }

    /** Determine the minimum number of arguments a callable can take
     * using the same trick. */
    template<typename Fn,
             typename ...Args,
             typename = decltype(std::declval<Fn>()(std::declval<Args>()...))>
    HALIDE_ALWAYS_INLINE
    static int num_args(int, Fn &&, Args...) {
        return (int)(sizeof...(Args));
    }

    /** The recursive version is only enabled up to a recursion limit
     * of 256. This catches callables that aren't callable with any
     * number of ints. */
    template<typename Fn,
             typename ...Args>
    HALIDE_ALWAYS_INLINE
    static int num_args(double, Fn &&f, Args... args) {
        static_assert(sizeof...(args) <= 256,
                      "Callable passed to for_each_element must accept either a const int *,"
                      " or up to 256 ints. No such operator found. Expect infinite template recursion.");
        return num_args(0, std::forward<Fn>(f), 0, args...);
    }

    /** A version where the callable takes a position array instead,
     * with compile-time recursion on the dimensionality.  This
     * overload is preferred to the one below using the same int vs
     * double trick as above, but is impossible once d hits -1 using
     * std::enable_if. */
    template<int d,
             typename Fn,
             typename = typename std::enable_if<(d >= 0)>::type>
    HALIDE_ALWAYS_INLINE
    static void for_each_element_array_helper(int, const for_each_element_task_dim *t, Fn &&f, int *pos) {
        for (pos[d] = t[d].min; pos[d] <= t[d].max; pos[d]++) {
            for_each_element_array_helper<d - 1>(0, t, std::forward<Fn>(f), pos);
        }
    }

    /** Base case for recursion above. */
    template<int d,
             typename Fn,
             typename = typename std::enable_if<(d < 0)>::type>
    HALIDE_ALWAYS_INLINE
    static void for_each_element_array_helper(double, const for_each_element_task_dim *t, Fn &&f, int *pos) {
        f(pos);
    }

    /** A run-time-recursive version (instead of
     * compile-time-recursive) that requires the callable to take a
     * pointer to a position array instead. Dispatches to the
     * compile-time-recursive version once the dimensionality gets
     * small. */
    template<typename Fn>
    static void for_each_element_array(int d, const for_each_element_task_dim *t, Fn &&f, int *pos) {
        if (d == -1) {
            f(pos);
        } else if (d == 0) {
            // Once the dimensionality gets small enough, dispatch to
            // a compile-time-recursive version for better codegen of
            // the inner loops.
            for_each_element_array_helper<0, Fn>(0, t, std::forward<Fn>(f), pos);
        } else if (d == 1) {
            for_each_element_array_helper<1, Fn>(0, t, std::forward<Fn>(f), pos);
        } else if (d == 2) {
            for_each_element_array_helper<2, Fn>(0, t, std::forward<Fn>(f), pos);
        } else if (d == 3) {
            for_each_element_array_helper<3, Fn>(0, t, std::forward<Fn>(f), pos);
        } else {
            for (pos[d] = t[d].min; pos[d] <= t[d].max; pos[d]++) {
                for_each_element_array(d - 1, t, std::forward<Fn>(f), pos);
            }
        }
    }

    /** We now have two overloads for for_each_element. This one
     * triggers if the callable takes a const int *.
     */
    template<typename Fn,
             typename = decltype(std::declval<Fn>()((const int *)nullptr))>
    static void for_each_element(int, int dims, const for_each_element_task_dim *t, Fn &&f, int check = 0) {
        int *pos = (int *)HALIDE_ALLOCA(dims * sizeof(int));
        for_each_element_array(dims - 1, t, std::forward<Fn>(f), pos);
    }

    /** This one triggers otherwise. It treats the callable as
     * something that takes some number of ints. */
    template<typename Fn>
    HALIDE_ALWAYS_INLINE
    static void for_each_element(double, int dims, const for_each_element_task_dim *t, Fn &&f) {
        int args = num_args(0, std::forward<Fn>(f));
        assert(dims >= args);
        for_each_element_variadic(0, args - 1, t, std::forward<Fn>(f));
    }
public:

    /** Call a function at each site in a buffer. This is likely to be
     * much slower than using Halide code to populate a buffer, but is
     * convenient for tests. If the function has more arguments than the
     * buffer has dimensions, the remaining arguments will be zero. If it
     * has fewer arguments than the buffer has dimensions then the last
     * few dimensions of the buffer are not iterated over. For example,
     * the following code exploits this to set a floating point RGB image
     * to red:

     \code
     Buffer<float, 3> im(100, 100, 3);
     im.for_each_element([&](int x, int y) {
         im(x, y, 0) = 1.0f;
         im(x, y, 1) = 0.0f;
         im(x, y, 2) = 0.0f:
     });
     \endcode

     * The compiled code is equivalent to writing the a nested for loop,
     * and compilers are capable of optimizing it in the same way.
     *
     * If the callable can be called with an int * as the sole argument,
     * that version is called instead. Each location in the buffer is
     * passed to it in a coordinate array. This version is higher-overhead
     * than the variadic version, but is useful for writing generic code
     * that accepts buffers of arbitrary dimensionality. For example, the
     * following sets the value at all sites in an arbitrary-dimensional
     * buffer to their first coordinate:

     \code
     im.for_each_element([&](const int *pos) {im(pos) = pos[0];});
     \endcode

     * It is also possible to use for_each_element to iterate over entire
     * rows or columns by cropping the buffer to a single column or row
     * respectively and iterating over elements of the result. For example,
     * to set the diagonal of the image to 1 by iterating over the columns:

     \code
     Buffer<float, 3> im(100, 100, 3);
         im.sliced(1, 0).for_each_element([&](int x, int c) {
         im(x, x, c) = 1.0f;
     });
     \endcode

     * Or, assuming the memory layout is known to be dense per row, one can
     * memset each row of an image like so:

     \code
     Buffer<float, 3> im(100, 100, 3);
     im.sliced(0, 0).for_each_element([&](int y, int c) {
         memset(&im(0, y, c), 0, sizeof(float) * im.width());
     });
     \endcode

    */
    template<typename Fn>
    void for_each_element(Fn &&f) const {
        for_each_element_task_dim *t =
            (for_each_element_task_dim *)HALIDE_ALLOCA(dimensions() * sizeof(for_each_element_task_dim));
        for (int i = 0; i < dimensions(); i++) {
            t[i].min = dim(i).min();
            t[i].max = dim(i).max();
        }
        for_each_element(0, dimensions(), t, std::forward<Fn>(f));
    }

private:
    template<typename Fn>
    struct FillHelper {
        Fn f;
        Buffer<T, D> *buf;

        template<typename... Args,
                 typename = decltype(std::declval<Fn>()(std::declval<Args>()...))>
        void operator()(Args... args) {
            (*buf)(args...) = f(args...);
        }

        FillHelper(Fn &&f, Buffer<T, D> *buf) : f(std::forward<Fn>(f)), buf(buf) {}
    };

public:
    /** Fill a buffer by evaluating a callable at every site. The
     * callable should look much like a callable passed to
     * for_each_element, but it should return the value that should be
     * stored to the coordinate corresponding to the arguments. */
    template<typename Fn,
             typename = typename std::enable_if<!std::is_arithmetic<typename std::decay<Fn>::type>::value>::type>
    void fill(Fn &&f) {
        // We'll go via for_each_element. We need a variadic wrapper lambda.
        FillHelper<Fn> wrapper(std::forward<Fn>(f), this);
        for_each_element(wrapper);
    }

    /** Check if an input buffer passed extern stage is a querying
     * bounds. Compared to doing the host pointer check directly,
     * this both adds clarity to code and will facilitate moving to
     * another representation for bounds query arguments. */
    bool is_bounds_query() {
        return buf.is_bounds_query();
    }

};

}  // namespace Runtime
}  // namespace Halide

#undef HALIDE_ALLOCA

#endif  // HALIDE_RUNTIME_IMAGE_H
#ifndef HALIDE_DEVICE_INTERFACE_H
#define HALIDE_DEVICE_INTERFACE_H

/** \file
 * Methods for managing device allocations when jitting
 */

#ifndef HALIDE_TARGET_H
#define HALIDE_TARGET_H

/** \file
 * Defines the structure that describes a Halide target.
 */

#include <stdint.h>
#include <bitset>
#include <string>


namespace Halide {

/** A struct representing a target machine and os to generate code for. */
struct Target {
    /** The operating system used by the target. Determines which
     * system calls to generate.
     * Corresponds to os_name_map in Target.cpp. */
    enum OS {OSUnknown = 0, Linux, Windows, OSX, Android, IOS, QuRT, NoOS} os;

    /** The architecture used by the target. Determines the
     * instruction set to use.
     * Corresponds to arch_name_map in Target.cpp. */
    enum Arch {ArchUnknown = 0, X86, ARM, MIPS, Hexagon, POWERPC} arch;

    /** The bit-width of the target machine. Must be 0 for unknown, or 32 or 64. */
    int bits;

    /** Optional features a target can have.
     * Corresponds to feature_name_map in Target.cpp.
     * See definitions in HalideRuntime.h for full information.
     */
    enum Feature {
        JIT = halide_target_feature_jit,
        Debug = halide_target_feature_debug,
        NoAsserts = halide_target_feature_no_asserts,
        NoBoundsQuery = halide_target_feature_no_bounds_query,
        SSE41 = halide_target_feature_sse41,
        AVX = halide_target_feature_avx,
        AVX2 = halide_target_feature_avx2,
        FMA = halide_target_feature_fma,
        FMA4 = halide_target_feature_fma4,
        F16C = halide_target_feature_f16c,
        ARMv7s = halide_target_feature_armv7s,
        NoNEON = halide_target_feature_no_neon,
        VSX = halide_target_feature_vsx,
        POWER_ARCH_2_07 = halide_target_feature_power_arch_2_07,
        CUDA = halide_target_feature_cuda,
        CUDACapability30 = halide_target_feature_cuda_capability30,
        CUDACapability32 = halide_target_feature_cuda_capability32,
        CUDACapability35 = halide_target_feature_cuda_capability35,
        CUDACapability50 = halide_target_feature_cuda_capability50,
        CUDACapability61 = halide_target_feature_cuda_capability61,
        OpenCL = halide_target_feature_opencl,
        CLDoubles = halide_target_feature_cl_doubles,
        OpenGL = halide_target_feature_opengl,
        OpenGLCompute = halide_target_feature_openglcompute,
        UserContext = halide_target_feature_user_context,
        Matlab = halide_target_feature_matlab,
        Profile = halide_target_feature_profile,
        NoRuntime = halide_target_feature_no_runtime,
        Metal = halide_target_feature_metal,
        MinGW = halide_target_feature_mingw,
        CPlusPlusMangling = halide_target_feature_c_plus_plus_mangling,
        LargeBuffers = halide_target_feature_large_buffers,
        HVX_64 = halide_target_feature_hvx_64,
        HVX_128 = halide_target_feature_hvx_128,
        HVX_v62 = halide_target_feature_hvx_v62,
        HVX_v65 = halide_target_feature_hvx_v65,
        HVX_v66 = halide_target_feature_hvx_v66,
        HVX_shared_object = halide_target_feature_hvx_use_shared_object,
        FuzzFloatStores = halide_target_feature_fuzz_float_stores,
        SoftFloatABI = halide_target_feature_soft_float_abi,
        MSAN = halide_target_feature_msan,
        AVX512 = halide_target_feature_avx512,
        AVX512_KNL = halide_target_feature_avx512_knl,
        AVX512_Skylake = halide_target_feature_avx512_skylake,
        AVX512_Cannonlake = halide_target_feature_avx512_cannonlake,
        TraceLoads = halide_target_feature_trace_loads,
        TraceStores = halide_target_feature_trace_stores,
        TraceRealizations = halide_target_feature_trace_realizations,
        FeatureEnd = halide_target_feature_end
    };
    Target() : os(OSUnknown), arch(ArchUnknown), bits(0) {}
    Target(OS o, Arch a, int b, std::vector<Feature> initial_features = std::vector<Feature>())
        : os(o), arch(a), bits(b) {
        for (size_t i = 0; i < initial_features.size(); i++) {
            set_feature(initial_features[i]);
        }
    }

    /** Given a string of the form used in HL_TARGET
     * (e.g. "x86-64-avx"), construct the Target it specifies. Note
     * that this always starts with the result of get_host_target(),
     * replacing only the parts found in the target string, so if you
     * omit (say) an OS specification, the host OS will be used
     * instead. An empty string is exactly equivalent to
     * get_host_target().
     *
     * Invalid target strings will fail with a user_error.
     */
    // @{
    EXPORT explicit Target(const std::string &s);
    EXPORT explicit Target(const char *s);
    // @}

    /** Check if a target string is valid. */
    EXPORT static bool validate_target_string(const std::string &s);

    void set_feature(Feature f, bool value = true) {
        if (f == FeatureEnd) return;
        user_assert(f < FeatureEnd) << "Invalid Target feature.\n";
        features.set(f, value);
    }

    void set_features(std::vector<Feature> features_to_set, bool value = true) {
        for (Feature f : features_to_set) {
            set_feature(f, value);
        }
    }

    bool has_feature(Feature f) const {
        if (f == FeatureEnd) return true;
        user_assert(f < FeatureEnd) << "Invalid Target feature.\n";
        return features[f];
    }

    bool features_any_of(std::vector<Feature> test_features) const {
        for (Feature f : test_features) {
            if (has_feature(f)) {
                return true;
            }
        }
        return false;
    }

    bool features_all_of(std::vector<Feature> test_features) const {
        for (Feature f : test_features) {
            if (!has_feature(f)) {
                return false;
            }
        }
        return true;
    }

    /** Return a copy of the target with the given feature set.
     * This is convenient when enabling certain features (e.g. NoBoundsQuery)
     * in an initialization list, where the target to be mutated may be
     * a const reference. */
    Target with_feature(Feature f) const {
        Target copy = *this;
        copy.set_feature(f);
        return copy;
    }

    /** Return a copy of the target with the given feature cleared.
     * This is convenient when disabling certain features (e.g. NoBoundsQuery)
     * in an initialization list, where the target to be mutated may be
     * a const reference. */
    Target without_feature(Feature f) const {
        Target copy = *this;
        copy.set_feature(f, false);
        return copy;
    }

    /** Is a fully feature GPU compute runtime enabled? I.e. is
     * Func::gpu_tile and similar going to work? Currently includes
     * CUDA, OpenCL, and Metal. We do not include OpenGL, because it
     * is not capable of gpgpu, and is not scheduled via
     * Func::gpu_tile.
     * TODO: Should OpenGLCompute be included here? */
    bool has_gpu_feature() const {
        return has_feature(CUDA) || has_feature(OpenCL) || has_feature(Metal);
    }

    /** Does this target allow using a certain type. Generally all
     * types except 64-bit float and int/uint should be supported by
     * all backends.
     *
     * It is likely better to call the version below which takes a DeviceAPI.
     */
    bool supports_type(const Type &t) const {
        if (t.bits() == 64) {
            if (t.is_float()) {
                return !has_feature(Metal) &&
                       (!has_feature(Target::OpenCL) || has_feature(Target::CLDoubles));
            } else {
                return !has_feature(Metal);
            }
        }
        return true;
    }

    /** Does this target allow using a certain type on a certain device.
     * This is the prefered version of this routine.
     */
    EXPORT bool supports_type(const Type &t, DeviceAPI device) const;

    /** Returns whether a particular device API can be used with this
     * Target. */
    EXPORT bool supports_device_api(DeviceAPI api) const;

    bool operator==(const Target &other) const {
      return os == other.os &&
          arch == other.arch &&
          bits == other.bits &&
          features == other.features;
    }

    bool operator!=(const Target &other) const {
      return !(*this == other);
    }

    /** Convert the Target into a string form that can be reconstituted
     * by merge_string(), which will always be of the form
     *
     *   arch-bits-os-feature1-feature2...featureN.
     *
     * Note that is guaranteed that Target(t1.to_string()) == t1,
     * but not that Target(s).to_string() == s (since there can be
     * multiple strings that parse to the same Target)...
     * *unless* t1 contains 'unknown' fields (in which case you'll get a string
     * that can't be parsed, which is intentional).
     */
    EXPORT std::string to_string() const;

    /** Given a data type, return an estimate of the "natural" vector size
     * for that data type when compiling for this Target. */
    int natural_vector_size(Halide::Type t) const {
        user_assert(os != OSUnknown && arch != ArchUnknown && bits != 0)
            << "natural_vector_size cannot be used on a Target with Unknown values.\n";

        const bool is_integer = t.is_int() || t.is_uint();
        const int data_size = t.bytes();

        if (arch == Target::Hexagon) {
            if (is_integer) {
                // HVX is either 64 or 128 *byte* vector size.
                if (has_feature(Halide::Target::HVX_128)) {
                    return 128 / data_size;
                } else if (has_feature(Halide::Target::HVX_64)) {
                    return 64 / data_size;
                } else {
                    user_error << "Target uses hexagon arch without hvx_128 or hvx_64 set.\n";
                    return 0;
                }
            } else {
                // HVX does not have vector float instructions.
                return 1;
            }
        } else if (arch == Target::X86) {
            if (is_integer && (has_feature(Halide::Target::AVX512_Skylake) ||
                               has_feature(Halide::Target::AVX512_Cannonlake))) {
                // AVX512BW exists on Skylake and Cannonlake
                return 64 / data_size;
            } else if (t.is_float() && (has_feature(Halide::Target::AVX512) ||
                                        has_feature(Halide::Target::AVX512_KNL) ||
                                        has_feature(Halide::Target::AVX512_Skylake) ||
                                        has_feature(Halide::Target::AVX512_Cannonlake))) {
                // AVX512F is on all AVX512 architectures
                return 64 / data_size;
            } else if (has_feature(Halide::Target::AVX2)) {
                // AVX2 uses 256-bit vectors for everything.
                return 32 / data_size;
            } else if (!is_integer && has_feature(Halide::Target::AVX)) {
                // AVX 1 has 256-bit vectors for float, but not for
                // integer instructions.
                return 32 / data_size;
            } else {
                // SSE was all 128-bit. We ignore MMX.
                return 16 / data_size;
            }
        } else {
            // Assume 128-bit vectors on other targets.
            return 16 / data_size;
        }
    }

    /** Given a data type, return an estimate of the "natural" vector size
     * for that data type when compiling for this Target. */
    template <typename data_t>
    int natural_vector_size() const {
        return natural_vector_size(type_of<data_t>());
    }

    /** Return true iff 64 bits and has_feature(LargeBuffers). */
    bool has_large_buffers() const {
        return bits == 64 && has_feature(LargeBuffers);
    }

    /** Return the maximum buffer size in bytes supported on this
     * Target. This is 2^31 - 1 except on 64-bit targets when the LargeBuffers
     * feature is enabled, which expands the maximum to 2^63 - 1. */
    int64_t maximum_buffer_size() const {
        if (has_large_buffers()) {
            return (((uint64_t)1) << 63) - 1;
        } else {
            return (((uint64_t)1) << 31) - 1;
        }
    }

    /** Was libHalide compiled with support for this target? */
    EXPORT bool supported() const;

private:
    /** A bitmask that stores the active features. */
    std::bitset<FeatureEnd> features;
};

/** Return the target corresponding to the host machine. */
EXPORT Target get_host_target();

/** Return the target that Halide will use. If HL_TARGET is set it
 * uses that. Otherwise calls \ref get_host_target */
EXPORT Target get_target_from_environment();

/** Return the target that Halide will use for jit-compilation. If
 * HL_JIT_TARGET is set it uses that. Otherwise calls \ref
 * get_host_target. Throws an error if the architecture, bit width,
 * and OS of the target do not match the host target, so this is only
 * useful for controlling the feature set. */
EXPORT Target get_jit_target_from_environment();

/** Get the Target feature corresponding to a DeviceAPI. For device
 * apis that do not correspond to any single target feature, returns
 * Target::FeatureEnd */
EXPORT Target::Feature target_feature_for_device_api(DeviceAPI api);

namespace Internal {

EXPORT void target_test();

}

}

#endif

namespace Halide {

/** Get the appropriate halide_device_interface_t * for a
 * target. Corresponds to the device interface that would be used for
 * DeviceAPI::Default_GPU. Creates a GPU runtime module for the target
 * if necessary. Returns nullptr if no device APIs are enabled in the
 * target. */
EXPORT const halide_device_interface_t *get_default_device_interface_for_target(const Target &t);

/** Gets the appropriate halide_device_interface_t * for a
 * DeviceAPI. Returns null if that device API is not enabled in the
 * target, or if the argument is None or Host. */
EXPORT const halide_device_interface_t *get_device_interface_for_device_api(const DeviceAPI &d,
                                                                            const Target &t = get_jit_target_from_environment());

/** Get the specific DeviceAPI that Halide would select when presented
 * with DeviceAPI::Default_GPU for a given target. If no suitable api
 * is enabled in the target, returns DeviceAPI::Host. */
EXPORT DeviceAPI get_default_device_api_for_target(const Target &t);

namespace Internal {
/** Get an Expr which evaluates to the device interface for the given device api at runtime. */
Expr make_device_interface_call(DeviceAPI device_api);
}

}

#endif

namespace Halide {

template<typename T = void> class Buffer;

namespace Internal {

struct BufferContents {
    mutable RefCount ref_count;
    std::string name;
    Runtime::Buffer<> buf;
};

EXPORT Expr buffer_accessor(const Buffer<> &buf, const std::vector<Expr> &args);

template<typename ...Args>
struct all_ints_and_optional_name : std::false_type {};

template<typename First, typename ...Rest>
struct all_ints_and_optional_name<First, Rest...> :
        meta_and<std::is_convertible<First, int>,
                 all_ints_and_optional_name<Rest...>> {};

template<typename T> struct all_ints_and_optional_name<T> :
        meta_or<std::is_convertible<T, std::string>,
                std::is_convertible<T, int>> {};

template<> struct all_ints_and_optional_name<> : std::true_type {};

template<typename T,
         typename = typename std::enable_if<!std::is_convertible<T, std::string>::value>::type>
std::string get_name_from_end_of_parameter_pack(T&&) {
    return "";
}

inline std::string get_name_from_end_of_parameter_pack(const std::string &n) {
    return n;
}

inline std::string get_name_from_end_of_parameter_pack() {
    return "";
}

template<typename First,
         typename Second,
         typename ...Args>
std::string get_name_from_end_of_parameter_pack(First first, Second second, Args&&... rest) {
    return get_name_from_end_of_parameter_pack(second, std::forward<Args>(rest)...);
}

inline void get_shape_from_start_of_parameter_pack_helper(std::vector<int> &, const std::string &) {
}

inline void get_shape_from_start_of_parameter_pack_helper(std::vector<int> &) {
}

template<typename ...Args>
void get_shape_from_start_of_parameter_pack_helper(std::vector<int> &result, int x, Args&&... rest) {
    result.push_back(x);
    get_shape_from_start_of_parameter_pack_helper(result, std::forward<Args>(rest)...);
}


template<typename ...Args>
std::vector<int> get_shape_from_start_of_parameter_pack(Args&&... args) {
    std::vector<int> result;
    get_shape_from_start_of_parameter_pack_helper(result, std::forward<Args>(args)...);
    return result;
}

template<typename T, typename T2>
using add_const_if_T_is_const = typename std::conditional<std::is_const<T>::value, const T2, T2>::type;

}

/** A Halide::Buffer is a named shared reference to a
 * Halide::Runtime::Buffer.
 *
 * A Buffer<T1> can refer to a Buffer<T2> if T1 is const whenever T2
 * is const, and either T1 = T2 or T1 is void. A Buffer<void> can
 * refer to any Buffer of any non-const type, and the default
 * template parameter is T = void.
 */
template<typename T>
class Buffer {
    Internal::IntrusivePtr<Internal::BufferContents> contents;

    template<typename T2> friend class Buffer;

    template<typename T2>
    static void assert_can_convert_from(const Buffer<T2> &other) {
        Runtime::Buffer<T>::assert_can_convert_from(*(other.get()));
    }

public:

    typedef T ElemType;

    /** Make a null Buffer, which points to no Runtime::Buffer */
    Buffer() {}

    /** Make a Buffer from a Buffer of a different type */
    template<typename T2>
    Buffer(const Buffer<T2> &other) :
        contents(other.contents) {
        assert_can_convert_from(other);
    }

    /** Move construct from a Buffer of a different type */
    template<typename T2>
    Buffer(Buffer<T2> &&other) {
        assert_can_convert_from(other);
        contents = std::move(other.contents);
    }

    /** Construct a Buffer that captures and owns an rvalue Runtime::Buffer */
    template<int D>
    Buffer(Runtime::Buffer<T, D> &&buf, const std::string &name = "") :
        contents(new Internal::BufferContents) {
        contents->buf = std::move(buf);
        if (name.empty()) {
            contents->name = Internal::make_entity_name(this, "Halide::Buffer<?", 'b');
        } else {
            contents->name = name;
        }
    }

    /** Constructors that match Runtime::Buffer with two differences:
     * 1) They take a Type instead of a halide_type_t
     * 2) There is an optional last string argument that gives the buffer a specific name
     */
    // @{
    template<typename ...Args,
             typename = typename std::enable_if<Internal::all_ints_and_optional_name<Args...>::value>::type>
    explicit Buffer(Type t,
                    int first, Args... rest) :
        Buffer(Runtime::Buffer<T>(t, Internal::get_shape_from_start_of_parameter_pack(first, rest...)),
               Internal::get_name_from_end_of_parameter_pack(rest...)) {}

    explicit Buffer(const halide_buffer_t &buf,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(buf), name) {}

    explicit Buffer(const buffer_t &buf,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(buf), name) {}

    template<typename ...Args,
             typename = typename std::enable_if<Internal::all_ints_and_optional_name<Args...>::value>::type>
    explicit Buffer(int first, Args... rest) :
        Buffer(Runtime::Buffer<T>(Internal::get_shape_from_start_of_parameter_pack(first, rest...)),
               Internal::get_name_from_end_of_parameter_pack(rest...)) {}

    explicit Buffer(Type t,
                    const std::vector<int> &sizes,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(t, sizes), name) {}

    explicit Buffer(const std::vector<int> &sizes,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(sizes), name) {}

    template<typename Array, size_t N>
    explicit Buffer(Array (&vals)[N],
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(vals), name) {}

    template<typename ...Args,
             typename = typename std::enable_if<Internal::all_ints_and_optional_name<Args...>::value>::type>
    explicit Buffer(Type t,
                    Internal::add_const_if_T_is_const<T, void> *data,
                    int first, Args&&... rest) :
        Buffer(Runtime::Buffer<T>(t, data, Internal::get_shape_from_start_of_parameter_pack(first, rest...)),
               Internal::get_name_from_end_of_parameter_pack(rest...)) {}

    template<typename ...Args,
             typename = typename std::enable_if<Internal::all_ints_and_optional_name<Args...>::value>::type>
    explicit Buffer(T *data,
                    int first, Args&&... rest) :
        Buffer(Runtime::Buffer<T>(data, Internal::get_shape_from_start_of_parameter_pack(first, rest...)),
               Internal::get_name_from_end_of_parameter_pack(rest...)) {}

    explicit Buffer(T *data,
                    const std::vector<int> &sizes,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(data, sizes), name) {}

    explicit Buffer(Type t,
                    Internal::add_const_if_T_is_const<T, void> *data,
                    const std::vector<int> &sizes,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(t, data, sizes), name) {}

    explicit Buffer(Type t,
                    Internal::add_const_if_T_is_const<T, void> *data,
                    int d,
                    const halide_dimension_t *shape,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(t, data, d, shape), name) {}

    explicit Buffer(T *data,
                    int d,
                    const halide_dimension_t *shape,
                    const std::string &name = "") :
        Buffer(Runtime::Buffer<T>(data, d, shape), name) {}


    static Buffer<T> make_scalar(const std::string &name = "") {
        return Buffer<T>(Runtime::Buffer<T>::make_scalar(), name);
    }

    static Buffer<> make_scalar(Type t, const std::string &name = "") {
        return Buffer<>(Runtime::Buffer<>::make_scalar(t), name);
    }

    static Buffer<T> make_interleaved(int width, int height, int channels, const std::string &name = "") {
        return Buffer<T>(Runtime::Buffer<T>::make_interleaved(width, height, channels),
                        name);
    }

    static Buffer<> make_interleaved(Type t, int width, int height, int channels, const std::string &name = "") {
        return Buffer<>(Runtime::Buffer<>::make_interleaved(t, width, height, channels),
                        name);
    }

    static Buffer<T> make_interleaved(T *data, int width, int height, int channels, const std::string &name = "") {
        return Buffer<T>(Runtime::Buffer<T>::make_interleaved(data, width, height, channels),
                         name);
    }

    static Buffer<Internal::add_const_if_T_is_const<T, void>>
    make_interleaved(Type t, T *data, int width, int height, int channels, const std::string &name = "") {
        using T2 = Internal::add_const_if_T_is_const<T, void>;
        return Buffer<T2>(Runtime::Buffer<T2>::make_interleaved(t, data, width, height, channels),
                          name);
    }

    template<typename T2>
    static Buffer<T> make_with_shape_of(Buffer<T2> src,
                                        void *(*allocate_fn)(size_t) = nullptr,
                                        void (*deallocate_fn)(void *) = nullptr,
                                        const std::string &name = "") {
        return Buffer<T>(Runtime::Buffer<T>::make_with_shape_of(*src.get(), allocate_fn, deallocate_fn),
                         name);
    }

    template<typename T2>
    static Buffer<T> make_with_shape_of(const Runtime::Buffer<T2> &src,
                                        void *(*allocate_fn)(size_t) = nullptr,
                                        void (*deallocate_fn)(void *) = nullptr,
                                        const std::string &name = "") {
        return Buffer<T>(Runtime::Buffer<T>::make_with_shape_of(src, allocate_fn, deallocate_fn),
                         name);
    }
    // @}

    /** Buffers are optionally named. */
    // @{
    void set_name(const std::string &n) {
        contents->name = n;
    }

    const std::string &name() const {
        return contents->name;
    }
    // @}

    /** Check if two Buffer objects point to the same underlying Buffer */
    template<typename T2>
    bool same_as(const Buffer<T2> &other) {
        return (const void *)(contents.get()) == (const void *)(other.contents.get());
    }

    /** Check if this Buffer refers to an existing
     * Buffer. Default-constructed Buffer objects do not refer to any
     * existing Buffer. */
    bool defined() const {
        return contents.defined();
    }

    /** Get a pointer to the underlying Runtime::Buffer */
    // @{
    Runtime::Buffer<T> *get() {
        // It's already type-checked, so no need to use as<T>.
        return (Runtime::Buffer<T> *)(&contents->buf);
    }
    const Runtime::Buffer<T> *get() const {
        return (const Runtime::Buffer<T> *)(&contents->buf);
    }
    // @}

public:

    // We forward numerous methods from the underlying Buffer
#define HALIDE_BUFFER_FORWARD_CONST(method)                             \
    template<typename ...Args>                                          \
    auto method(Args&&... args) const ->                                \
        decltype(std::declval<const Runtime::Buffer<T>>().method(std::forward<Args>(args)...)) { \
        user_assert(defined()) << "Undefined buffer calling const method " #method "\n";         \
        return get()->method(std::forward<Args>(args)...);                                       \
    }

#define HALIDE_BUFFER_FORWARD(method)                                   \
    template<typename ...Args>                                          \
    auto method(Args&&... args) ->                                      \
        decltype(std::declval<Runtime::Buffer<T>>().method(std::forward<Args>(args)...)) { \
        user_assert(defined()) << "Undefined buffer calling method " #method "\n";         \
        return get()->method(std::forward<Args>(args)...);                                 \
    }

    /** Does the same thing as the equivalent Halide::Runtime::Buffer method */
    // @{
    HALIDE_BUFFER_FORWARD(raw_buffer)
    HALIDE_BUFFER_FORWARD_CONST(raw_buffer)
    HALIDE_BUFFER_FORWARD_CONST(dimensions)
    HALIDE_BUFFER_FORWARD_CONST(dim)
    HALIDE_BUFFER_FORWARD_CONST(width)
    HALIDE_BUFFER_FORWARD_CONST(height)
    HALIDE_BUFFER_FORWARD_CONST(channels)
    HALIDE_BUFFER_FORWARD_CONST(min)
    HALIDE_BUFFER_FORWARD_CONST(extent)
    HALIDE_BUFFER_FORWARD_CONST(stride)
    HALIDE_BUFFER_FORWARD_CONST(left)
    HALIDE_BUFFER_FORWARD_CONST(right)
    HALIDE_BUFFER_FORWARD_CONST(top)
    HALIDE_BUFFER_FORWARD_CONST(bottom)
    HALIDE_BUFFER_FORWARD_CONST(number_of_elements)
    HALIDE_BUFFER_FORWARD_CONST(size_in_bytes)
    HALIDE_BUFFER_FORWARD_CONST(begin)
    HALIDE_BUFFER_FORWARD_CONST(end)
    HALIDE_BUFFER_FORWARD(data)
    HALIDE_BUFFER_FORWARD_CONST(data)
    HALIDE_BUFFER_FORWARD_CONST(contains)
    HALIDE_BUFFER_FORWARD(crop)
    HALIDE_BUFFER_FORWARD(slice)
    HALIDE_BUFFER_FORWARD_CONST(sliced)
    HALIDE_BUFFER_FORWARD(embed)
    HALIDE_BUFFER_FORWARD_CONST(embedded)
    HALIDE_BUFFER_FORWARD(set_min)
    HALIDE_BUFFER_FORWARD(translate)
    HALIDE_BUFFER_FORWARD(transpose)
    HALIDE_BUFFER_FORWARD(add_dimension)
    HALIDE_BUFFER_FORWARD(copy_to_host)
    HALIDE_BUFFER_FORWARD(copy_to_device)
    HALIDE_BUFFER_FORWARD_CONST(has_device_allocation)
    HALIDE_BUFFER_FORWARD_CONST(host_dirty)
    HALIDE_BUFFER_FORWARD_CONST(device_dirty)
    HALIDE_BUFFER_FORWARD(set_host_dirty)
    HALIDE_BUFFER_FORWARD(set_device_dirty)
    HALIDE_BUFFER_FORWARD(device_sync)
    HALIDE_BUFFER_FORWARD(device_malloc)
    HALIDE_BUFFER_FORWARD(device_wrap_native)
    HALIDE_BUFFER_FORWARD(device_detach_native)
    HALIDE_BUFFER_FORWARD(allocate)
    HALIDE_BUFFER_FORWARD(deallocate)
    HALIDE_BUFFER_FORWARD(device_deallocate)
    HALIDE_BUFFER_FORWARD(device_free)
    HALIDE_BUFFER_FORWARD(fill)
    HALIDE_BUFFER_FORWARD_CONST(for_each_element)

#undef HALIDE_BUFFER_FORWARD
#undef HALIDE_BUFFER_FORWARD_CONST

    template<typename Fn, typename ...Args>
    void for_each_value(Fn &&f, Args... other_buffers) {
        return get()->for_each_value(std::forward<Fn>(f), (*std::forward<Args>(other_buffers).get())...);
    }

    static constexpr bool has_static_halide_type = Runtime::Buffer<T>::has_static_halide_type;

    static halide_type_t static_halide_type() {
        return Runtime::Buffer<T>::static_halide_type();
    }

    template<typename T2>
    static bool can_convert_from(const Buffer<T2> &other) {
        return Halide::Runtime::Buffer<T>::can_convert_from(*other.get());
    }

    Type type() const {
        return contents->buf.type();
    }

    template<typename T2>
    Buffer<T2> as() const {
        return Buffer<T2>(*this);
    }

    Buffer<T> copy() const {
        return Buffer<T>(std::move(contents->buf.copy()));
    }

    template<typename T2>
    void copy_from(const Buffer<T2> &other) {
        contents->buf.copy_from(*other.get());
    }

    template<typename ...Args>
    auto operator()(int first, Args&&... args) ->
        decltype(std::declval<Runtime::Buffer<T>>()(first, std::forward<Args>(args)...)) {
        return (*get())(first, std::forward<Args>(args)...);
    }

    template<typename ...Args>
    auto operator()(int first, Args&&... args) const ->
        decltype(std::declval<const Runtime::Buffer<T>>()(first, std::forward<Args>(args)...)) {
        return (*get())(first, std::forward<Args>(args)...);
    }

    auto operator()(const int *pos) ->
        decltype(std::declval<Runtime::Buffer<T>>()(pos)) {
        return (*get())(pos);
    }

    auto operator()(const int *pos) const ->
        decltype(std::declval<const Runtime::Buffer<T>>()(pos)) {
        return (*get())(pos);
    }

    auto operator()() ->
        decltype(std::declval<Runtime::Buffer<T>>()()) {
        return (*get())();
    }

    auto operator()() const ->
        decltype(std::declval<const Runtime::Buffer<T>>()()) {
        return (*get())();
    }
    // @}

    /** Make an Expr that loads from this concrete buffer at a computed coordinate. */
    // @{
    template<typename ...Args>
    Expr operator()(Expr first, Args... rest) const {
        std::vector<Expr> args = {first, rest...};
        return (*this)(args);
    };

    template<typename ...Args>
    Expr operator()(const std::vector<Expr> &args) const {
        return buffer_accessor(Buffer<>(*this), args);
    };
    // @}


    /** Copy to the GPU, using the device API that is the default for the given Target. */
    int copy_to_device(const Target &t = get_jit_target_from_environment()) {
        return contents->buf.copy_to_device(get_default_device_interface_for_target(t));
    }

    /** Copy to the GPU, using the given device API */
    int copy_to_device(const DeviceAPI &d, const Target &t = get_jit_target_from_environment()) {
        return contents->buf.copy_to_device(get_device_interface_for_device_api(d, t));
    }

    /** Allocate on the GPU, using the device API that is the default for the given Target. */
    int device_malloc(const Target &t = get_jit_target_from_environment()) {
        return contents->buf.device_malloc(get_default_device_interface_for_target(t));
    }

    /** Allocate storage on the GPU, using the given device API */
    int device_malloc(const DeviceAPI &d, const Target &t = get_jit_target_from_environment()) {
        return contents->buf.device_malloc(get_device_interface_for_device_api(d, t));
    }

    /** Wrap a native handle, using the given device API.
     * It is a bad idea to pass DeviceAPI::Default_GPU to this routine
     * as the handle argument must match the API that the default
     * resolves to and it is clearer and more reliable to pass the
     * resolved DeviceAPI explicitly. */
    int device_wrap_native(const DeviceAPI &d, uint64_t handle, const Target &t = get_jit_target_from_environment()) {
        return contents->buf.device_wrap_native(get_device_interface_for_device_api(d, t), handle);
    }

};

}

#endif

namespace Halide {

class OutputImageParam;

namespace Internal {

class Constrainable;
struct ParameterContents;

/** A reference-counted handle to a parameter to a halide
 * pipeline. May be a scalar parameter or a buffer */
class Parameter {
    IntrusivePtr<ParameterContents> contents;

    void check_defined() const;
    void check_is_buffer() const;
    void check_is_scalar() const;
    void check_dim_ok(int dim) const;

public:
    /** Construct a new undefined handle */
    EXPORT Parameter();

    /** Construct a new parameter of the given type. If the second
     * argument is true, this is a buffer parameter of the given
     * dimensionality, otherwise, it is a scalar parameter (and the
     * dimensionality should be zero). The parameter will be given a
     * unique auto-generated name. */
    EXPORT Parameter(Type t, bool is_buffer, int dimensions);

    /** Construct a new parameter of the given type with name given by
     * the third argument. If the second argument is true, this is a
     * buffer parameter, otherwise, it is a scalar parameter. The
     * third argument gives the dimensionality of the buffer
     * parameter. It should be zero for scalar parameters. If the
     * fifth argument is true, the the name being passed in was
     * explicitly specified (as opposed to autogenerated). If the
     * sixth argument is true, the Parameter is registered in the global
     * ObjectInstanceRegistry. */
    EXPORT Parameter(Type t, bool is_buffer, int dimensions,
                     const std::string &name, bool is_explicit_name = false,
                     bool register_instance = true, bool is_bound_before_lowering = false);

    /** Copy ctor, operator=, and dtor, needed for ObjectRegistry accounting. */
    EXPORT Parameter(const Parameter&);
    EXPORT Parameter& operator=(const Parameter&);
    EXPORT ~Parameter();

    /** Get the type of this parameter */
    EXPORT Type type() const;

    /** Get the dimensionality of this parameter. Zero for scalars. */
    EXPORT int dimensions() const;

    /** Get the name of this parameter */
    EXPORT const std::string &name() const;

    /** Return true iff the name was explicitly specified */
    EXPORT bool is_explicit_name() const;

    /** Return true iff this Parameter is expected to be replaced with a
     * constant at the start of lowering, and thus should not be used to
     * infer arguments */
    EXPORT bool is_bound_before_lowering() const;

    /** Does this parameter refer to a buffer/image? */
    EXPORT bool is_buffer() const;

    /** If the parameter is a scalar parameter, get its currently
     * bound value. Only relevant when jitting */
    template<typename T>
    NO_INLINE T get_scalar() const {
        // Allow get_scalar<uint64_t>() for all Handle types
        user_assert(type() == type_of<T>() || (type().is_handle() && type_of<T>() == UInt(64)))
            << "Can't get Param<" << type()
            << "> as scalar of type " << type_of<T>() << "\n";
        return *((const T *)(get_scalar_address()));
    }

    /** This returns the current value of get_scalar<type()>()
     * as an Expr. */
    EXPORT Expr get_scalar_expr() const;

    /** If the parameter is a scalar parameter, set its current
     * value. Only relevant when jitting */
    template<typename T>
    NO_INLINE void set_scalar(T val) {
        // Allow set_scalar<uint64_t>() for all Handle types
        user_assert(type() == type_of<T>() || (type().is_handle() && type_of<T>() == UInt(64)))
            << "Can't set Param<" << type()
            << "> to scalar of type " << type_of<T>() << "\n";
        *((T *)(get_scalar_address())) = val;
    }

    /** If the parameter is a buffer parameter, get its currently
     * bound buffer. Only relevant when jitting */
    EXPORT Buffer<> get_buffer() const;

    /** If the parameter is a buffer parameter, set its current
     * value. Only relevant when jitting */
    EXPORT void set_buffer(Buffer<> b);

    /** Get the pointer to the current value of the scalar
     * parameter. For a given parameter, this address will never
     * change. Only relevant when jitting. */

    EXPORT void *get_scalar_address() const;

    /** Tests if this handle is the same as another handle */
    EXPORT bool same_as(const Parameter &other) const;

    /** Tests if this handle is non-nullptr */
    EXPORT bool defined() const;

    /** Get and set constraints for the min, extent, stride, and estimates on
     * the min/extent. */
    //@{
    EXPORT void set_min_constraint(int dim, Expr e);
    EXPORT void set_extent_constraint(int dim, Expr e);
    EXPORT void set_stride_constraint(int dim, Expr e);
    EXPORT void set_min_constraint_estimate(int dim, Expr min);
    EXPORT void set_extent_constraint_estimate(int dim, Expr extent);
    EXPORT void set_host_alignment(int bytes);
    EXPORT Expr min_constraint(int dim) const;
    EXPORT Expr extent_constraint(int dim) const;
    EXPORT Expr stride_constraint(int dim) const;
    EXPORT Expr min_constraint_estimate(int dim) const;
    EXPORT Expr extent_constraint_estimate(int dim) const;
    EXPORT int host_alignment() const;
    //@}

    /** Get and set constraints for scalar parameters. These are used
     * directly by Param, so they must be exported. */
    // @{
    EXPORT void set_min_value(Expr e);
    EXPORT Expr get_min_value() const;
    EXPORT void set_max_value(Expr e);
    EXPORT Expr get_max_value() const;
    EXPORT void set_estimate(Expr e);
    EXPORT Expr get_estimate() const;
    // @}
};

class Dimension {
public:
    /** Get an expression representing the minimum coordinates of this image
     * parameter in the given dimension. */
    EXPORT Expr min() const;

    /** Get an expression representing the extent of this image
     * parameter in the given dimension */
    EXPORT Expr extent() const;

    /** Get an expression representing the maximum coordinates of
     * this image parameter in the given dimension. */
    EXPORT Expr max() const;

    /** Get an expression representing the stride of this image in the
     * given dimension */
    EXPORT Expr stride() const;

    /** Get the estimate of the minimum coordinate of this image parameter
     * in the given dimension. Return an undefined expr if the estimate is
     * never specified. */
    EXPORT Expr min_estimate() const;

    /** Get the estimate of the extent of this image parameter in the given
     * dimension. Return an undefined expr if the estimate is never specified. */
    EXPORT Expr extent_estimate() const;

    /** Set the min in a given dimension to equal the given
     * expression. Setting the mins to zero may simplify some
     * addressing math. */
    EXPORT Dimension set_min(Expr e);

    /** Set the extent in a given dimension to equal the given
     * expression. Images passed in that fail this check will generate
     * a runtime error. Returns a reference to the ImageParam so that
     * these calls may be chained.
     *
     * This may help the compiler generate better
     * code. E.g:
     \code
     im.dim(0).set_extent(100);
     \endcode
     * tells the compiler that dimension zero must be of extent 100,
     * which may result in simplification of boundary checks. The
     * value can be an arbitrary expression:
     \code
     im.dim(0).set_extent(im.dim(1).extent());
     \endcode
     * declares that im is a square image (of unknown size), whereas:
     \code
     im.dim(0).set_extent((im.dim(0).extent()/32)*32);
     \endcode
     * tells the compiler that the extent is a multiple of 32. */
    EXPORT Dimension set_extent(Expr e);

    /** Set the stride in a given dimension to equal the given
     * value. This is particularly helpful to set when
     * vectorizing. Known strides for the vectorized dimension
     * generate better code. */
    EXPORT Dimension set_stride(Expr e);

    /** Set the min and extent in one call. */
    EXPORT Dimension set_bounds(Expr min, Expr extent);

    /** Set the estimate of the min in a given dimension to equal the given
     * expression. This value is only used by the auto-scheduler. */
    EXPORT Dimension set_min_estimate(Expr e);

    /** Set the estimate of the extent in a given dimension to equal the given
     * expression. This value is only used by the auto-scheduler. */
    EXPORT Dimension set_extent_estimate(Expr e);

    /** Set the min and extent estimates in one call. These values are only
     * used by the auto-scheduler. */
    EXPORT Dimension set_bounds_estimate(Expr min, Expr extent);

    /** Get a different dimension of the same buffer */
    // @{
    EXPORT Dimension dim(int i);
    EXPORT const Dimension dim(int i) const;
    // @}

private:
    friend class ::Halide::OutputImageParam;
    friend class Constrainable;

    /** Construct a Dimension representing dimension d of some
     * Internal::Parameter p. Only friends may construct
     * these. */
    EXPORT Dimension(const Internal::Parameter &p, int d);

    /** Only friends may copy these, too. This prevents
     * users removing constness by making a non-const copy. */
    Dimension(const Dimension &) = default;

    Parameter param;
    int d;
};


/** Validate arguments to a call to a func, image or imageparam. */
void check_call_arg_types(const std::string &name, std::vector<Expr> *args, int dims);

}
}

#endif
#ifndef HALIDE_SCHEDULE_H
#define HALIDE_SCHEDULE_H

/** \file
 * Defines the internal representation of the schedule for a function
 */


#include <map>

namespace Halide {

class Func;
template <typename T> class ScheduleParam;
struct VarOrRVar;

namespace Internal {
class Function;
struct FunctionContents;
struct LoopLevelContents;
class ScheduleParamBase;
}  // namespace Internal

/** Different ways to handle a tail case in a split when the
 * factor does not provably divide the extent. */
enum class TailStrategy {
    /** Round up the extent to be a multiple of the split
     * factor. Not legal for RVars, as it would change the meaning
     * of the algorithm. Pros: generates the simplest, fastest
     * code. Cons: if used on a stage that reads from the input or
     * writes to the output, constrains the input or output size
     * to be a multiple of the split factor. */
    RoundUp,

    /** Guard the inner loop with an if statement that prevents
     * evaluation beyond the original extent. Always legal. The if
     * statement is treated like a boundary condition, and
     * factored out into a loop epilogue if possible. Pros: no
     * redundant re-evaluation; does not constrain input our
     * output sizes. Cons: increases code size due to separate
     * tail-case handling; vectorization will scalarize in the tail
     * case to handle the if statement. */
    GuardWithIf,

    /** Prevent evaluation beyond the original extent by shifting
     * the tail case inwards, re-evaluating some points near the
     * end. Only legal for pure variables in pure definitions. If
     * the inner loop is very simple, the tail case is treated
     * like a boundary condition and factored out into an
     * epilogue.
     *
     * This is a good trade-off between several factors. Like
     * RoundUp, it supports vectorization well, because the inner
     * loop is always a fixed size with no data-dependent
     * branching. It increases code size slightly for inner loops
     * due to the epilogue handling, but not for outer loops
     * (e.g. loops over tiles). If used on a stage that reads from
     * an input or writes to an output, this stategy only requires
     * that the input/output extent be at least the split factor,
     * instead of a multiple of the split factor as with RoundUp. */
    ShiftInwards,

    /** For pure definitions use ShiftInwards. For pure vars in
     * update definitions use RoundUp. For RVars in update
     * definitions use GuardWithIf. */
    Auto
};

/** Different ways to handle accesses outside the original extents in a prefetch. */
enum class PrefetchBoundStrategy {
    /** Clamp the prefetched exprs by intersecting the prefetched region with
     * the original extents. This may make the exprs of the prefetched region
     * more complicated. */
    Clamp,

    /** Guard the prefetch with if-guards that ignores the prefetch if
     * any of the prefetched region ever goes beyond the original extents
     * (i.e. all or nothing). */
    GuardWithIf,

    /** Leave the prefetched exprs as are (no if-guards around the prefetch
     * and no intersecting with the original extents). This makes the prefetch
     * exprs simpler but this may cause prefetching of region outside the original
     * extents. This is good if prefetch won't fault when accessing region
     * outside the original extents. */
    NonFaulting
};

/** A reference to a site in a Halide statement at the top of the
 * body of a particular for loop. Evaluating a region of a halide
 * function is done by generating a loop nest that spans its
 * dimensions. We schedule the inputs to that function by
 * recursively injecting realizations for them at particular sites
 * in this loop nest. A LoopLevel identifies such a site. */
class LoopLevel {
    template <typename T> friend class ScheduleParam;
    friend class ::Halide::Internal::ScheduleParamBase;

    Internal::IntrusivePtr<Internal::LoopLevelContents> contents;

    explicit LoopLevel(Internal::IntrusivePtr<Internal::LoopLevelContents> c) : contents(c) {}
    EXPORT LoopLevel(const std::string &func_name, const std::string &var_name, bool is_rvar);

    /** Mutate our contents to match the contents of 'other'. This is a potentially
     * dangerous operation to do if you aren't careful, and exists solely to make
     * ScheduleParam<LoopLevel> easy to implement; hence its private status. */
    EXPORT void copy_from(const LoopLevel &other);

public:
    /** Identify the loop nest corresponding to some dimension of some function */
    // @{
    EXPORT LoopLevel(Internal::Function f, VarOrRVar v);
    EXPORT LoopLevel(Func f, VarOrRVar v);
    // @}

    /** Construct an undefined LoopLevel. Calling any method on an undefined
     * LoopLevel (other than defined() or operator==) will assert. */
    LoopLevel() = default;

    /** Return true iff the LoopLevel is defined. */
    EXPORT bool defined() const;

    /** Return the Func name. Asserts if the LoopLevel is_root() or is_inline(). */
    EXPORT std::string func() const;

    /** Return the VarOrRVar. Asserts if the LoopLevel is_root() or is_inline(). */
    EXPORT VarOrRVar var() const;

    /** inlined is a special LoopLevel value that implies
     * that a function should be inlined away. */
    EXPORT static LoopLevel inlined();

    /** Test if a loop level corresponds to inlining the function */
    EXPORT bool is_inline() const;

    /** root is a special LoopLevel value which represents the
     * location outside of all for loops */
    EXPORT static LoopLevel root();

    /** Test if a loop level is 'root', which describes the site
     * outside of all for loops */
    EXPORT bool is_root() const;

    /** Return a string of the form func.var -- note that this is safe
     * to call for root or inline LoopLevels. */
    EXPORT std::string to_string() const;

    /** Compare this loop level against the variable name of a for
     * loop, to see if this loop level refers to the site
     * immediately inside this loop. */
    EXPORT bool match(const std::string &loop) const;

    EXPORT bool match(const LoopLevel &other) const;

    /** Check if two loop levels are exactly the same. */
    EXPORT bool operator==(const LoopLevel &other) const;

    bool operator!=(const LoopLevel &other) const { return !(*this == other); }
};

namespace Internal {

class IRMutator;
struct ReductionVariable;

struct Split {
    std::string old_var, outer, inner;
    Expr factor;
    bool exact; // Is it required that the factor divides the extent
                // of the old var. True for splits of RVars. Forces
                // tail strategy to be GuardWithIf.
    TailStrategy tail;

    enum SplitType {SplitVar = 0, RenameVar, FuseVars, PurifyRVar};

    // If split_type is Rename, then this is just a renaming of the
    // old_var to the outer and not a split. The inner var should
    // be ignored, and factor should be one. Renames are kept in
    // the same list as splits so that ordering between them is
    // respected.

    // If split type is Purify, this replaces the old_var RVar to
    // the outer Var. The inner var should be ignored, and factor
    // should be one.

    // If split_type is Fuse, then this does the opposite of a
    // split, it joins the outer and inner into the old_var.
    SplitType split_type;

    bool is_rename() const {return split_type == RenameVar;}
    bool is_split() const {return split_type == SplitVar;}
    bool is_fuse() const {return split_type == FuseVars;}
    bool is_purify() const {return split_type == PurifyRVar;}
};

struct Dim {
    std::string var;
    ForType for_type;
    DeviceAPI device_api;

    enum Type {PureVar = 0, PureRVar, ImpureRVar};
    Type dim_type;

    bool is_pure() const {return (dim_type == PureVar) || (dim_type == PureRVar);}
    bool is_rvar() const {return (dim_type == PureRVar) || (dim_type == ImpureRVar);}
    bool is_parallel() const {
        return (for_type == ForType::Parallel ||
                for_type == ForType::GPUBlock ||
                for_type == ForType::GPUThread);
    }
};

struct Bound {
    std::string var;
    Expr min, extent, modulus, remainder;
};

struct StorageDim {
    std::string var;
    Expr alignment;
    Expr fold_factor;
    bool fold_forward;
};

struct PrefetchDirective {
    std::string name;
    std::string var;
    Expr offset;
    PrefetchBoundStrategy strategy;
    // If it's a prefetch load from an image parameter, this points to that.
    Parameter param;
};

struct FuncScheduleContents;
struct StageScheduleContents;
struct FunctionContents;

/** A schedule for a Function of a Halide pipeline. This schedule is
 * applied to all stages of the Function. Right now this interface is
 * basically a struct, offering mutable access to its innards.
 * In the future it may become more encapsulated. */
class FuncSchedule {
    IntrusivePtr<FuncScheduleContents> contents;

public:

    FuncSchedule(IntrusivePtr<FuncScheduleContents> c) : contents(c) {}
    FuncSchedule(const FuncSchedule &other) : contents(other.contents) {}
    EXPORT FuncSchedule();

    /** Return a deep copy of this FuncSchedule. It recursively deep copies all
     * called functions, schedules, specializations, and reduction domains. This
     * method takes a map of <old FunctionContents, deep-copied version> as input
     * and would use the deep-copied FunctionContents from the map if exists
     * instead of creating a new deep-copy to avoid creating deep-copies of the
     * same FunctionContents multiple times.
     */
    EXPORT FuncSchedule deep_copy(
        std::map<FunctionPtr, FunctionPtr> &copied_map) const;

    /** This flag is set to true if the schedule is memoized. */
    // @{
    bool &memoized();
    bool memoized() const;
    // @}

    /** The list and order of dimensions used to store this
     * function. The first dimension in the vector corresponds to the
     * innermost dimension for storage (i.e. which dimension is
     * tightly packed in memory) */
    // @{
    const std::vector<StorageDim> &storage_dims() const;
    std::vector<StorageDim> &storage_dims();
    // @}

    /** You may explicitly bound some of the dimensions of a function,
     * or constrain them to lie on multiples of a given factor. See
     * \ref Func::bound and \ref Func::align_bounds */
    // @{
    const std::vector<Bound> &bounds() const;
    std::vector<Bound> &bounds();
    // @}

    /** You may explicitly specify an estimate of some of the function
     * dimensions. See \ref Func::estimate */
    // @{
    const std::vector<Bound> &estimates() const;
    std::vector<Bound> &estimates();
    // @}

    /** Mark calls of a function by 'f' to be replaced with its identity
     * wrapper or clone during the lowering stage. If the string 'f' is empty,
     * it means replace all calls to the function by all other functions
     * (excluding itself) in the pipeline with the global identity wrapper.
     * See \ref Func::in and \ref Func::clone for more details. */
    // @{
    const std::map<std::string, Internal::FunctionPtr> &wrappers() const;
    std::map<std::string, Internal::FunctionPtr> &wrappers();
    EXPORT void add_wrapper(const std::string &f,
                            const Internal::FunctionPtr &wrapper);
    // @}

    /** At what sites should we inject the allocation and the
     * computation of this function? The store_level must be outside
     * of or equal to the compute_level. If the compute_level is
     * inline, the store_level is meaningless. See \ref Func::store_at
     * and \ref Func::compute_at */
    // @{
    const LoopLevel &store_level() const;
    const LoopLevel &compute_level() const;
    LoopLevel &store_level();
    LoopLevel &compute_level();
    // @}

    /** Pass an IRVisitor through to all Exprs referenced in the
     * Schedule. */
    void accept(IRVisitor *) const;

    /** Pass an IRMutator through to all Exprs referenced in the
     * Schedule. */
    void mutate(IRMutator *);
};


/** A schedule for a single stage of a Halide pipeline. Right now this
 * interface is basically a struct, offering mutable access to its
 * innards. In the future it may become more encapsulated. */
class StageSchedule {
    IntrusivePtr<StageScheduleContents> contents;

public:

    StageSchedule(IntrusivePtr<StageScheduleContents> c) : contents(c) {}
    StageSchedule(const StageSchedule &other) : contents(other.contents) {}
    EXPORT StageSchedule();

    /** Return a copy of this StageSchedule. */
    EXPORT StageSchedule get_copy() const;

    /** This flag is set to true if the dims list has been manipulated
     * by the user (or if a ScheduleHandle was created that could have
     * been used to manipulate it). It controls the warning that
     * occurs if you schedule the vars of the pure step but not the
     * update steps. */
    // @{
    bool &touched();
    bool touched() const;
    // @}

    /** RVars of reduction domain associated with this schedule if there is any. */
    // @{
    EXPORT const std::vector<ReductionVariable> &rvars() const;
    std::vector<ReductionVariable> &rvars();
    // @}

    /** The traversal of the domain of a function can have some of its
     * dimensions split into sub-dimensions. See \ref Func::split */
    // @{
    const std::vector<Split> &splits() const;
    std::vector<Split> &splits();
    // @}

    /** The list and ordering of dimensions used to evaluate this
     * function, after all splits have taken place. The first
     * dimension in the vector corresponds to the innermost for loop,
     * and the last is the outermost. Also specifies what type of for
     * loop to use for each dimension. Does not specify the bounds on
     * each dimension. These get inferred from how the function is
     * used, what the splits are, and any optional bounds in the list below. */
    // @{
    const std::vector<Dim> &dims() const;
    std::vector<Dim> &dims();
    // @}

    /** You may perform prefetching in some of the dimensions of a
     * function. See \ref Func::prefetch */
    // @{
    const std::vector<PrefetchDirective> &prefetches() const;
    std::vector<PrefetchDirective> &prefetches();
    // @}

    /** Are race conditions permitted? */
    // @{
    bool allow_race_conditions() const;
    bool &allow_race_conditions();
    // @}

    /** Pass an IRVisitor through to all Exprs referenced in the
     * Schedule. */
    void accept(IRVisitor *) const;

    /** Pass an IRMutator through to all Exprs referenced in the
     * Schedule. */
    void mutate(IRMutator *);
};

}
}

#endif
#ifndef HALIDE_REDUCTION_H
#define HALIDE_REDUCTION_H

/** \file
 * Defines internal classes related to Reduction Domains
 */


namespace Halide {
namespace Internal {

class IRMutator;

/** A single named dimension of a reduction domain */
struct ReductionVariable {
    std::string var;
    Expr min, extent;

    /** This lets you use a ReductionVariable as a key in a map of the form
     * map<ReductionVariable, Foo, ReductionVariable::Compare> */
    struct Compare {
        bool operator()(const ReductionVariable &a, const ReductionVariable &b) const {
            return a.var < b.var;
        }
    };
};

struct ReductionDomainContents;

/** A reference-counted handle on a reduction domain, which is just a
 * vector of ReductionVariable. */
class ReductionDomain {
    IntrusivePtr<ReductionDomainContents> contents;
public:
    /** This lets you use a ReductionDomain as a key in a map of the form
     * map<ReductionDomain, Foo, ReductionDomain::Compare> */
    struct Compare {
        bool operator()(const ReductionDomain &a, const ReductionDomain &b) const {
            internal_assert(a.contents.defined() && b.contents.defined());
            return a.contents < b.contents;
        }
    };

    /** Construct a new nullptr reduction domain */
    ReductionDomain() : contents(nullptr) {}

    /** Construct a reduction domain that spans the outer product of
     * all values of the given ReductionVariable in scanline order,
     * with the start of the vector being innermost, and the end of
     * the vector being outermost. */
    EXPORT ReductionDomain(const std::vector<ReductionVariable> &domain);

    /** Return a deep copy of this ReductionDomain. */
    EXPORT ReductionDomain deep_copy() const;

    /** Is this handle non-nullptr */
    bool defined() const {
        return contents.defined();
    }

    /** Tests for equality of reference. Only one reduction domain is
     * allowed per reduction function, and this is used to verify
     * that */
    bool same_as(const ReductionDomain &other) const {
        return contents.same_as(other.contents);
    }

    /** Immutable access to the reduction variables. */
    EXPORT const std::vector<ReductionVariable> &domain() const;

    /** Add predicate to the reduction domain. See \ref RDom::where
     * for more details. */
    EXPORT void where(Expr predicate);

    /** Return the predicate defined on this reducation demain. */
    EXPORT Expr predicate() const;

    /** Set the predicate, replacing any previously set predicate. */
    EXPORT void set_predicate(Expr);

    /** Split predicate into vector of ANDs. If there is no predicate (i.e. all
     * iteration domain in this reduction domain is valid), this returns an
     * empty vector. */
    EXPORT std::vector<Expr> split_predicate() const;

    /** Mark RDom as frozen, which means it cannot accept new predicates. An
     * RDom is frozen once it is used in a Func's update definition. */
    EXPORT void freeze();

    /** Check if a RDom has been frozen. If so, it is an error to add new
     * predicates. */
    EXPORT bool frozen() const;

    /** Pass an IRVisitor through to all Exprs referenced in the
     * ReductionDomain. */
    void accept(IRVisitor *) const;

    /** Pass an IRMutator through to all Exprs referenced in the
     * ReductionDomain. */
    void mutate(IRMutator *);
};

EXPORT void split_predicate_test();

}
}

#endif
#ifndef HALIDE_DEFINITION_H
#define HALIDE_DEFINITION_H

/** \file
 * Defines the internal representation of a halide function's definition and related classes
 */


#include <map>

namespace Halide {

namespace Internal {
struct DefinitionContents;
struct FunctionContents;
}

namespace Internal {

class IRVisitor;
class IRMutator;
struct Specialization;

/** A Function definition which can either represent a init or an update
 * definition. A function may have different definitions due to specialization,
 * which are stored in 'specializations' (Not possible from the front-end, but
 * some scheduling directives may potentially cause this divergence to occur).
 * Although init definition may have multiple values (RHS) per specialization, it
 * must have the same LHS (i.e. same pure dimension variables). The update
 * definition, on the other hand, may have different LHS/RHS per specialization.
 * Note that, while the Expr in LHS/RHS may be different across specializations,
 * they must have the same number of dimensions and the same pure dimensions.
 */
class Definition {

    IntrusivePtr<DefinitionContents> contents;

public:
    /** Construct a Definition from an existing DefinitionContents pointer. Must be non-null */
    EXPORT explicit Definition(const IntrusivePtr<DefinitionContents> &);

    /** Construct a Definition with the supplied args, values, and reduction domain. */
    EXPORT Definition(const std::vector<Expr> &args, const std::vector<Expr> &values,
                      const ReductionDomain &rdom, bool is_init);

    /** Construct an empty Definition. By default, it is a init definition. */
    EXPORT Definition();

    /** Return a copy of this Definition. */
    EXPORT Definition get_copy() const;

    /** Equality of identity */
    bool same_as(const Definition &other) const {
        return contents.same_as(other.contents);
    }

    /** Is this an init definition; otherwise it's an update definition */
    EXPORT bool is_init() const;

    /** Pass an IRVisitor through to all Exprs referenced in the
     * definition. */
    EXPORT void accept(IRVisitor *) const;

    /** Pass an IRMutator through to all Exprs referenced in the
     * definition. */
    EXPORT void mutate(IRMutator *);

    /** Get the default (no-specialization) arguments (left-hand-side) of the definition */
    // @{
    EXPORT const std::vector<Expr> &args() const;
    EXPORT std::vector<Expr> &args();
    // @}

    /** Get the default (no-specialization) right-hand-side of the definition */
    // @{
    EXPORT const std::vector<Expr> &values() const;
    EXPORT std::vector<Expr> &values();
    // @}

    /** Get the predicate on the definition */
    // @{
    EXPORT const Expr &predicate() const;
    EXPORT Expr &predicate();
    // @}

    /** Split predicate into vector of ANDs. If there is no predicate (i.e. this
     * definition is always valid), this returns an empty vector. */
    EXPORT std::vector<Expr> split_predicate() const;

    /** Get the default (no-specialization) stage-specific schedule associated
     * with this definition. */
    // @{
    EXPORT const StageSchedule &schedule() const;
    EXPORT StageSchedule &schedule();
    // @}

    /** You may create several specialized versions of a func with
     * different stage-specific schedules. They trigger when the condition is
     * true. See \ref Func::specialize */
    // @{
    EXPORT const std::vector<Specialization> &specializations() const;
    EXPORT std::vector<Specialization> &specializations();
    EXPORT const Specialization &add_specialization(Expr condition);
    // @}

};

struct Specialization {
    Expr condition;
    Definition definition;
    std::string failure_message;  // If non-empty, this specialization always assert-fails with this message.
};

}}

#endif

#include <map>

namespace Halide {

/** An argument to an extern-defined Func. May be a Function, Buffer,
 * ImageParam or Expr. */
struct ExternFuncArgument {
    enum ArgType {UndefinedArg = 0, FuncArg, BufferArg, ExprArg, ImageParamArg};
    ArgType arg_type;
    Internal::FunctionPtr func;
    Buffer<> buffer;
    Expr expr;
    Internal::Parameter image_param;

    ExternFuncArgument(Internal::FunctionPtr f): arg_type(FuncArg), func(f) {}

    template<typename T>
    ExternFuncArgument(Buffer<T> b): arg_type(BufferArg), buffer(b) {}
    ExternFuncArgument(Expr e): arg_type(ExprArg), expr(e) {}
    ExternFuncArgument(int e): arg_type(ExprArg), expr(e) {}
    ExternFuncArgument(float e): arg_type(ExprArg), expr(e) {}

    ExternFuncArgument(Internal::Parameter p) : arg_type(ImageParamArg), image_param(p) {
        // Scalar params come in via the Expr constructor.
        internal_assert(p.is_buffer());
    }
    ExternFuncArgument() : arg_type(UndefinedArg) {}

    bool is_func() const {return arg_type == FuncArg;}
    bool is_expr() const {return arg_type == ExprArg;}
    bool is_buffer() const {return arg_type == BufferArg;}
    bool is_image_param() const {return arg_type == ImageParamArg;}
    bool defined() const {return arg_type != UndefinedArg;}
};

/** An enum to specify calling convention for extern stages. */
enum class NameMangling {
    Default,   ///< Match whatever is specified in the Target
    C,         ///< No name mangling
    CPlusPlus, ///< C++ name mangling
};

namespace Internal {

struct Call;

/** A reference-counted handle to Halide's internal representation of
 * a function. Similar to a front-end Func object, but with no
 * syntactic sugar to help with definitions. */
class Function {

    FunctionPtr contents;

public:
    /** This lets you use a Function as a key in a map of the form
     * map<Function, Foo, Function::Compare> */
    struct Compare {
        bool operator()(const Function &a, const Function &b) const {
            internal_assert(a.contents.defined() && b.contents.defined());
            return a.contents < b.contents;
        }
    };

    /** Construct a new function with no definitions and no name. This
     * constructor only exists so that you can make vectors of
     * functions, etc.
     */
    EXPORT Function();

    /** Construct a new function with the given name */
    EXPORT explicit Function(const std::string &n);

    /** Construct a Function from an existing FunctionContents pointer. Must be non-null */
    EXPORT explicit Function(const FunctionPtr &);

    /** Get a handle on the halide function contents that this Function
     * represents. */
    FunctionPtr get_contents() const {
        return contents;
    }

    /** Deep copy this Function into 'copy'. It recursively deep copies all called
     * functions, schedules, update definitions, extern func arguments, specializations,
     * and reduction domains. This method does not deep-copy the Parameter objects.
     * This method also takes a map of <old Function, deep-copied version> as input
     * and would use the deep-copied Function from the map if exists instead of
     * creating a new deep-copy to avoid creating deep-copies of the same Function
     * multiple times. If 'name' is specified, copy's name will be set to that.
     */
    // @{
    EXPORT void deep_copy(FunctionPtr copy, std::map<FunctionPtr, FunctionPtr> &copied_map) const;
    EXPORT void deep_copy(std::string name, FunctionPtr copy,
                          std::map<FunctionPtr, FunctionPtr> &copied_map) const;
    // @}

    /** Add a pure definition to this function. It may not already
     * have a definition. All the free variables in 'value' must
     * appear in the args list. 'value' must not depend on any
     * reduction domain */
    EXPORT void define(const std::vector<std::string> &args, std::vector<Expr> values);

    /** Add an update definition to this function. It must already
     * have a pure definition but not an update definition, and the
     * length of args must match the length of args used in the pure
     * definition. 'value' must depend on some reduction domain, and
     * may contain variables from that domain as well as pure
     * variables. Any pure variables must also appear as Variables in
     * the args array, and they must have the same name as the pure
     * definition's argument in the same index. */
    EXPORT void define_update(const std::vector<Expr> &args, std::vector<Expr> values);

    /** Accept a visitor to visit all of the definitions and arguments
     * of this function. */
    EXPORT void accept(IRVisitor *visitor) const;

    /** Accept a mutator to mutator all of the definitions and
     * arguments of this function. */
    EXPORT void mutate(IRMutator *mutator);

    /** Get the name of the function. */
    EXPORT const std::string &name() const;

    /** Get a mutable handle to the init definition. */
    EXPORT Definition &definition();

    /** Get the init definition. */
    EXPORT const Definition &definition() const;

    /** Get the pure arguments. */
    EXPORT const std::vector<std::string> args() const;

    /** Get the dimensionality. */
    EXPORT int dimensions() const;

    /** Get the number of outputs. */
    int outputs() const {
        return (int)output_types().size();
    }

    /** Get the types of the outputs. */
    EXPORT const std::vector<Type> &output_types() const;

    /** Get the right-hand-side of the pure definition. */
    EXPORT const std::vector<Expr> &values() const;

    /** Does this function have a pure definition? */
    EXPORT bool has_pure_definition() const;

    /** Does this function *only* have a pure definition? */
    bool is_pure() const {
        return (has_pure_definition() &&
                !has_update_definition() &&
                !has_extern_definition());
    }

    /** Is it legal to inline this function? */
    EXPORT bool can_be_inlined() const;

    /** Get a handle to the function-specific schedule for the purpose
     * of modifying it. */
    EXPORT FuncSchedule &schedule();

    /** Get a const handle to the function-specific schedule for inspecting it. */
    EXPORT const FuncSchedule &schedule() const;

    /** Get a handle on the output buffer used for setting constraints
     * on it. */
    EXPORT const std::vector<Parameter> &output_buffers() const;

    /** Get a mutable handle to the stage-specfic schedule for the update
     * stage. */
    EXPORT StageSchedule &update_schedule(int idx = 0);

    /** Get a mutable handle to this function's update definition at
     * index 'idx'. */
    EXPORT Definition &update(int idx = 0);

    /** Get a const reference to this function's update definition at
     * index 'idx'. */
    EXPORT const Definition &update(int idx = 0) const;

    /** Get a const reference to this function's update definitions. */
    EXPORT const std::vector<Definition> &updates() const;

    /** Does this function have an update definition? */
    EXPORT bool has_update_definition() const;

    /** Check if the function has an extern definition. */
    EXPORT bool has_extern_definition() const;

    /** Get the name mangling specified for the extern definition. */
    EXPORT NameMangling extern_definition_name_mangling() const;

    /** Make a call node to the extern definition. An error if the
     * function has no extern definition. */
    EXPORT Expr make_call_to_extern_definition(const std::vector<Expr> &args,
                                               const Target &t) const;

    /** Check if the extern function being called expects the legacy
     * buffer_t type. */
    EXPORT bool extern_definition_uses_old_buffer_t() const;

    /** Get the proxy Expr for the extern stage. This is an expression
     * known to have the same data access pattern as the extern
     * stage. It must touch at least all of the memory that the extern
     * stage does, though it is permissible for it to be conservative
     * and touch a superset. For most Functions, including those with
     * extern definitions, this will be an undefined Expr. */
    // @{
    EXPORT Expr extern_definition_proxy_expr() const;
    EXPORT Expr &extern_definition_proxy_expr();
    // @}

    /** Add an external definition of this Func. */
    EXPORT void define_extern(const std::string &function_name,
                              const std::vector<ExternFuncArgument> &args,
                              const std::vector<Type> &types,
                              int dimensionality,
                              NameMangling mangling,
                              DeviceAPI device_api,
                              bool uses_old_buffer_t);

    /** Retrive the arguments of the extern definition. */
    // @{
    EXPORT const std::vector<ExternFuncArgument> &extern_arguments() const;
    EXPORT std::vector<ExternFuncArgument> &extern_arguments();
    // @}

    /** Get the name of the extern function called for an extern
     * definition. */
    EXPORT const std::string &extern_function_name() const;

    /** Get the DeviceAPI declared for an extern function. */
    EXPORT DeviceAPI extern_function_device_api() const;

    /** Test for equality of identity. */
    bool same_as(const Function &other) const {
        return contents.same_as(other.contents);
    }

    /** Get a const handle to the debug filename. */
    EXPORT const std::string &debug_file() const;

    /** Get a handle to the debug filename. */
    EXPORT std::string &debug_file();

    /** Use an an extern argument to another function. */
    operator ExternFuncArgument() const {
        return ExternFuncArgument(contents);
    }

    /** Tracing calls and accessors, passed down from the Func
     * equivalents. */
    // @{
    EXPORT void trace_loads();
    EXPORT void trace_stores();
    EXPORT void trace_realizations();
    EXPORT bool is_tracing_loads() const;
    EXPORT bool is_tracing_stores() const;
    EXPORT bool is_tracing_realizations() const;
    // @}

    /** Mark function as frozen, which means it cannot accept new
     * definitions. */
    EXPORT void freeze();

    /** Check if a function has been frozen. If so, it is an error to
     * add new definitions. */
    EXPORT bool frozen() const;

    /** Make a new Function with the same lifetime as this one, and
     * return a strong reference to it. Useful to create Functions which
     * have circular references to this one - e.g. the wrappers
     * produced by Func::in. */
    Function new_function_in_same_group(const std::string &);

    /** Mark calls of this function by 'f' to be replaced with its wrapper
     * during the lowering stage. If the string 'f' is empty, it means replace
     * all calls to this function by all other functions (excluding itself) in
     * the pipeline with the wrapper. This will also freeze 'wrapper' to prevent
     * user from updating the values of the Function it wraps via the wrapper.
     * See \ref Func::in for more details. */
    // @{
    EXPORT void add_wrapper(const std::string &f, Function &wrapper);
    EXPORT const std::map<std::string, FunctionPtr> &wrappers() const;
    // @}

    /** Check if a Function is a trivial wrapper around another
     * Function, Buffer, or Parameter. Returns the Call node if it
     * is. Otherwise returns null.
     */
    EXPORT const Call *is_wrapper() const;

    /** Replace every call to Functions in 'substitutions' keys by all Exprs
     * referenced in this Function to call to their substitute Functions (i.e.
     * the corresponding values in 'substitutions' map). */
    // @{
    EXPORT Function &substitute_calls(const std::map<FunctionPtr, FunctionPtr> &substitutions);
    EXPORT Function &substitute_calls(const Function &orig, const Function &substitute);
    // @}

    /** Find all Vars that are placeholders for ScheduleParams and substitute in
     * the corresponding constant value. */
    EXPORT Function &substitute_schedule_param_exprs();
};

/** Deep copy an entire Function DAG. */
std::pair<std::vector<Function>, std::map<std::string, Function>> deep_copy(
    const std::vector<Function> &outputs,
    const std::map<std::string, Function> &env);

}}

#endif

namespace Halide {
namespace Internal {

extern bool always_upcast;
extern void set_always_upcast();

/** The actual IR nodes begin here. Remember that all the Expr
 * nodes also have a public "type" property */

/** Cast a node from one type to another. Can't change vector widths. */
struct Cast : public ExprNode<Cast> {
    Expr value;

    EXPORT static Expr make(Type t, Expr v);

    static const IRNodeType _node_type = IRNodeType::Cast;
};

/** The sum of two expressions */
struct Add : public ExprNode<Add> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::Add;
};

/** The difference of two expressions */
struct Sub : public ExprNode<Sub> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::Sub;
};

/** The product of two expressions */
struct Mul : public ExprNode<Mul> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::Mul;
};

/** The ratio of two expressions */
struct Div : public ExprNode<Div> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::Div;
};

/** The remainder of a / b. Mostly equivalent to '%' in C, except that
 * the result here is always positive. For floats, this is equivalent
 * to calling fmod. */
struct Mod : public ExprNode<Mod> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::Mod;
};

/** The lesser of two values. */
struct Min : public ExprNode<Min> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b);

    EXPORT static Expr make2(Expr a, Expr b, bool upcast);

    static const IRNodeType _node_type = IRNodeType::Min;
};

/** The greater of two values */
struct Max : public ExprNode<Max> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b);

    // Need a separate name because otherwise it breaks a call to FoldLeft
    EXPORT static Expr make2(Expr a, Expr b, bool upcast);

    static const IRNodeType _node_type = IRNodeType::Max;
};

/** Is the first expression equal to the second */
struct EQ : public ExprNode<EQ> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::EQ;
};

/** Is the first expression not equal to the second */
struct NE : public ExprNode<NE> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::NE;
};

/** Is the first expression less than the second. */
struct LT : public ExprNode<LT> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::LT;
};

/** Is the first expression less than or equal to the second. */
struct LE : public ExprNode<LE> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::LE;
};

/** Is the first expression greater than the second. */
struct GT : public ExprNode<GT> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::GT;
};

/** Is the first expression greater than or equal to the second. */
struct GE : public ExprNode<GE> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b, bool upcast = false);

    static const IRNodeType _node_type = IRNodeType::GE;
};

/** Logical and - are both expressions true */
struct And : public ExprNode<And> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b);

    static const IRNodeType _node_type = IRNodeType::And;
};

/** Logical or - is at least one of the expression true */
struct Or : public ExprNode<Or> {
    Expr a, b;

    EXPORT static Expr make(Expr a, Expr b);

    static const IRNodeType _node_type = IRNodeType::Or;
};

/** Logical not - true if the expression false */
struct Not : public ExprNode<Not> {
    Expr a;

    EXPORT static Expr make(Expr a);

    static const IRNodeType _node_type = IRNodeType::Not;
};

/** A ternary operator. Evalutes 'true_value' and 'false_value',
 * then selects between them based on 'condition'. Equivalent to
 * the ternary operator in C. */
struct Select : public ExprNode<Select> {
    Expr condition, true_value, false_value;

    EXPORT static Expr make(Expr condition, Expr true_value, Expr false_value);

    static const IRNodeType _node_type = IRNodeType::Select;
};

/** Load a value from a named symbol if predicate is true. The buffer
 * is treated as an array of the 'type' of this Load node. That is,
 * the buffer has no inherent type. The name may be the name of an
 * enclosing allocation, an input or output buffer, or any other
 * symbol of type Handle(). */
struct Load : public ExprNode<Load> {
    std::string name;

    Expr predicate, index;

    // If it's a load from an image argument or compiled-in constant
    // image, this will point to that
    Buffer<> image;

    // If it's a load from an image parameter, this points to that
    Parameter param;

    EXPORT static Expr make(Type type, const std::string &name,
                            Expr index, Buffer<> image,
                            Parameter param, Expr predicate);

    static const IRNodeType _node_type = IRNodeType::Load;
};

/** A linear ramp vector node. This is vector with 'lanes' elements,
 * where element i is 'base' + i*'stride'. This is a convenient way to
 * pass around vectors without busting them up into individual
 * elements. E.g. a dense vector load from a buffer can use a ramp
 * node with stride 1 as the index. */
struct Ramp : public ExprNode<Ramp> {
    Expr base, stride;
    int lanes;

    EXPORT static Expr make(Expr base, Expr stride, int lanes);

    static const IRNodeType _node_type = IRNodeType::Ramp;
};

/** A vector with 'lanes' elements, in which every element is
 * 'value'. This is a special case of the ramp node above, in which
 * the stride is zero. */
struct Broadcast : public ExprNode<Broadcast> {
    Expr value;
    int lanes;

    EXPORT static Expr make(Expr value, int lanes);

    static const IRNodeType _node_type = IRNodeType::Broadcast;
};

/** A let expression, like you might find in a functional
 * language. Within the expression \ref Let::body, instances of the Var
 * node \ref Let::name refer to \ref Let::value. */
struct Let : public ExprNode<Let> {
    std::string name;
    Expr value, body;

    EXPORT static Expr make(const std::string &name, Expr value, Expr body);

    static const IRNodeType _node_type = IRNodeType::Let;
};

/** The statement form of a let node. Within the statement 'body',
 * instances of the Var named 'name' refer to 'value' */
struct LetStmt : public StmtNode<LetStmt> {
    std::string name;
    Expr value;
    Stmt body;

    EXPORT static Stmt make(const std::string &name, Expr value, Stmt body);

    static const IRNodeType _node_type = IRNodeType::LetStmt;
};

/** If the 'condition' is false, then evaluate and return the message,
 * which should be a call to an error function. */
struct AssertStmt : public StmtNode<AssertStmt> {
    // if condition then val else error out with message
    Expr condition;
    Expr message;

    EXPORT static Stmt make(Expr condition, Expr message);

    static const IRNodeType _node_type = IRNodeType::AssertStmt;
};

/** This node is a helpful annotation to do with permissions. If 'is_produce' is
 * set to true, this represents a producer node which may also contain updates;
 * otherwise, this represents a consumer node. If the producer node contains
 * updates, the body of the node will be a block of 'produce' and 'update'
 * in that order. In a producer node, the access is read-write only (or write
 * only if it doesn't have updates). In a consumer node, the access is read-only.
 * None of this is actually enforced, the node is purely for informative purposes
 * to help out our analysis during lowering. For every unique ProducerConsumer,
 * there is an associated Realize node with the same name that creates the buffer
 * being read from or written to in the body of the ProducerConsumer.
 */
struct ProducerConsumer : public StmtNode<ProducerConsumer> {
    std::string name;
    bool is_producer;
    Stmt body;

    EXPORT static Stmt make(const std::string &name, bool is_producer, Stmt body);

    EXPORT static Stmt make_produce(const std::string &name, Stmt body);
    EXPORT static Stmt make_consume(const std::string &name, Stmt body);

    static const IRNodeType _node_type = IRNodeType::ProducerConsumer;
};

/** Store a 'value' to the buffer called 'name' at a given 'index' if
 * 'predicate' is true. The buffer is interpreted as an array of the
 * same type as 'value'. The name may be the name of an enclosing
 * Allocate node, an output buffer, or any other symbol of type
 * Handle(). */
struct Store : public StmtNode<Store> {
    std::string name;
    Expr predicate, value, index;
    // If it's a store to an output buffer, then this parameter points to it.
    Parameter param;

    EXPORT static Stmt make(const std::string &name, Expr value, Expr index,
                            Parameter param, Expr predicate);

    static const IRNodeType _node_type = IRNodeType::Store;
};

/** This defines the value of a function at a multi-dimensional
 * location. You should think of it as a store to a multi-dimensional
 * array. It gets lowered to a conventional Store node. The name must
 * correspond to an output buffer or the name of an enclosing Realize
 * node. */
struct Provide : public StmtNode<Provide> {
    std::string name;
    std::vector<Expr> values;
    std::vector<Expr> args;

    EXPORT static Stmt make(const std::string &name, const std::vector<Expr> &values, const std::vector<Expr> &args);

    static const IRNodeType _node_type = IRNodeType::Provide;
};

/** Allocate a scratch area called with the given name, type, and
 * size. The buffer lives for at most the duration of the body
 * statement, within which it is freed. It is an error for an allocate
 * node not to contain a free node of the same buffer. Allocation only
 * occurs if the condition evaluates to true. Within the body of the
 * allocation, defines a symbol with the given name and the type
 * Handle(). */
struct Allocate : public StmtNode<Allocate> {
    std::string name;
    Type type;
    std::vector<Expr> extents;
    Expr condition;

    // These override the code generator dependent malloc and free
    // equivalents if provided. If the new_expr succeeds, that is it
    // returns non-nullptr, the function named be free_function is
    // guaranteed to be called. The free function signature must match
    // that of the code generator dependent free (typically
    // halide_free). If free_function is left empty, code generator
    // default will be called.
    Expr new_expr;
    std::string free_function;
    Stmt body;

    EXPORT static Stmt make(const std::string &name, Type type, const std::vector<Expr> &extents,
                            Expr condition, Stmt body,
                            Expr new_expr = Expr(), const std::string &free_function = std::string());

    /** A routine to check if the extents are all constants, and if so verify
     * the total size is less than 2^31 - 1. If the result is constant, but
     * overflows, this routine asserts. This returns 0 if the extents are
     * not all constants; otherwise, it returns the total constant allocation
     * size. */
    EXPORT static int32_t constant_allocation_size(const std::vector<Expr> &extents, const std::string &name);
    EXPORT int32_t constant_allocation_size() const;

    static const IRNodeType _node_type = IRNodeType::Allocate;
};

/** Free the resources associated with the given buffer. */
struct Free : public StmtNode<Free> {
    std::string name;

    EXPORT static Stmt make(const std::string &name);

    static const IRNodeType _node_type = IRNodeType::Free;
};

/** A single-dimensional span. Includes all numbers between min and
 * (min + extent - 1) */
struct Range {
    Expr min, extent;
    Range() {}
    Range(Expr min, Expr extent) : min(min), extent(extent) {
        internal_assert(min.type() == extent.type()) << "Region min and extent must have same type\n";
    }
};

/** A multi-dimensional box. The outer product of the elements */
typedef std::vector<Range> Region;

/** Allocate a multi-dimensional buffer of the given type and
 * size. Create some scratch memory that will back the function 'name'
 * over the range specified in 'bounds'. The bounds are a vector of
 * (min, extent) pairs for each dimension. Allocation only occurs if
 * the condition evaluates to true.
 */
struct Realize : public StmtNode<Realize> {
    std::string name;
    std::vector<Type> types;
    Region bounds;
    Expr condition;
    Stmt body;

    EXPORT static Stmt make(const std::string &name, const std::vector<Type> &types, const Region &bounds, Expr condition, Stmt body);

    static const IRNodeType _node_type = IRNodeType::Realize;

};

/** A sequence of statements to be executed in-order. 'rest' may be
 * undefined. Used rest.defined() to find out. */
struct Block : public StmtNode<Block> {
    Stmt first, rest;

    EXPORT static Stmt make(Stmt first, Stmt rest);
    /** Construct zero or more Blocks to invoke a list of statements in order.
     * This method may not return a Block statement if stmts.size() <= 1. */
    EXPORT static Stmt make(const std::vector<Stmt> &stmts);

    static const IRNodeType _node_type = IRNodeType::Block;
};

/** An if-then-else block. 'else' may be undefined. */
struct IfThenElse : public StmtNode<IfThenElse> {
    Expr condition;
    Stmt then_case, else_case;

    EXPORT static Stmt make(Expr condition, Stmt then_case, Stmt else_case = Stmt());

    static const IRNodeType _node_type = IRNodeType::IfThenElse;
};

/** Evaluate and discard an expression, presumably because it has some side-effect. */
struct Evaluate : public StmtNode<Evaluate> {
    Expr value;

    EXPORT static Stmt make(Expr v);

    static const IRNodeType _node_type = IRNodeType::Evaluate;
};

/** A function call. This can represent a call to some extern function
 * (like sin), but it's also our multi-dimensional version of a Load,
 * so it can be a load from an input image, or a call to another
 * halide function. These two types of call nodes don't survive all
 * the way down to code generation - the lowering process converts
 * them to Load nodes. */
struct Call : public ExprNode<Call> {
    std::string name;
    std::vector<Expr> args;
    typedef enum {Image,        ///< A load from an input image
                  Extern,       ///< A call to an external C-ABI function, possibly with side-effects
                  ExternCPlusPlus, ///< A call to an external C-ABI function, possibly with side-effects
                  PureExtern,   ///< A call to a guaranteed-side-effect-free external function
                  Halide,       ///< A call to a Func
                  Intrinsic,    ///< A possibly-side-effecty compiler intrinsic, which has special handling during codegen
                  PureIntrinsic ///< A side-effect-free version of the above.
    } CallType;
    CallType call_type;

    // Halide uses calls internally to represent certain operations
    // (instead of IR nodes). These are matched by name. Note that
    // these are deliberately char* (rather than std::string) so that
    // they can be referenced at static-initialization time without
    // risking ambiguous initalization order; we use a typedef to simplify
    // declaration.
    typedef const char* const ConstString;
    EXPORT static ConstString debug_to_file,
        reinterpret,
        bitwise_and,
        bitwise_not,
        bitwise_xor,
        bitwise_or,
        shift_left,
        shift_right,
        address_of,
        abs,
        absd,
        rewrite_buffer,
        random,
        lerp,
        popcount,
        count_leading_zeros,
        count_trailing_zeros,
        undef,      
        return_second,
        if_then_else,
        glsl_texture_load,
        glsl_texture_store,
        glsl_varying,
        image_load,
        image_store,
        make_struct,
        stringify,
        memoize_expr,
        alloca,
        likely,
        likely_if_innermost,
        register_destructor,
        div_round_to_zero,
        mod_round_to_zero,
        call_cached_indirect_function,
        prefetch,
        signed_integer_overflow,
        indeterminate_expression,
        bool_to_mask,
        cast_mask,
        select_mask,
        extract_mask_element,
        require,
        size_of_halide_buffer_t;

    // We also declare some symbolic names for some of the runtime
    // functions that we want to construct Call nodes to here to avoid
    // magic string constants and the potential risk of typos.
    EXPORT static ConstString
        buffer_get_min,
        buffer_get_extent,
        buffer_get_stride,
        buffer_get_max,
        buffer_get_host,
        buffer_get_device,
        buffer_get_device_interface,
        buffer_get_shape,
        buffer_get_host_dirty,
        buffer_get_device_dirty,
        buffer_get_type_code,
        buffer_get_type_bits,
        buffer_get_type_lanes,
        buffer_set_host_dirty,
        buffer_set_device_dirty,
        buffer_is_bounds_query,
        buffer_init,
        buffer_init_from_buffer,
        buffer_crop,
        buffer_set_bounds,
        trace;

    // If it's a call to another halide function, this call node holds
    // a possibly-weak reference to that function.
    FunctionPtr func;

    // If that function has multiple values, which value does this
    // call node refer to?
    int value_index;

    // If it's a call to an image, this call nodes hold a
    // pointer to that image's buffer
    Buffer<> image;

    // If it's a call to an image parameter, this call node holds a
    // pointer to that
    Parameter param;

    EXPORT static Expr make(Type type, const std::string &name, const std::vector<Expr> &args, CallType call_type,
                            FunctionPtr func = FunctionPtr(), int value_index = 0,
                            Buffer<> image = Buffer<>(), Parameter param = Parameter());

    /** Convenience constructor for calls to other halide functions */
    EXPORT static Expr make(Function func, const std::vector<Expr> &args, int idx = 0);

    /** Convenience constructor for loads from concrete images */
    static Expr make(Buffer<> image, const std::vector<Expr> &args) {
        return make(image.type(), image.name(), args, Image, FunctionPtr(), 0, image, Parameter());
    }

    /** Convenience constructor for loads from images parameters */
    static Expr make(Parameter param, const std::vector<Expr> &args) {
        return make(param.type(), param.name(), args, Image, FunctionPtr(), 0, Buffer<>(), param);
    }

    /** Check if a call node is pure within a pipeline, meaning that
     * the same args always give the same result, and the calls can be
     * reordered, duplicated, unified, etc without changing the
     * meaning of anything. Not transitive - doesn't guarantee the
     * args themselves are pure. An example of a pure Call node is
     * sqrt. If in doubt, don't mark a Call node as pure. */
    bool is_pure() const {
        return (call_type == PureExtern ||
                call_type == Image ||
                call_type == PureIntrinsic);
    }

    bool is_intrinsic() const {
        return (call_type == Intrinsic ||
                call_type == PureIntrinsic);
    }

    bool is_intrinsic(ConstString intrin_name) const {
        return is_intrinsic() && name == intrin_name;
    }

    bool is_extern() const {
        return (call_type == Extern ||
                call_type == ExternCPlusPlus ||
                call_type == PureExtern);
    }

    static const IRNodeType _node_type = IRNodeType::Call;
};

/** A named variable. Might be a loop variable, function argument,
 * parameter, reduction variable, or something defined by a Let or
 * LetStmt node. */
struct Variable : public ExprNode<Variable> {
    std::string name;

    /** References to scalar parameters, or to the dimensions of buffer
     * parameters hang onto those expressions. */
    Parameter param;

    /** References to properties of literal image parameters. */
    Buffer<> image;

    /** Reduction variables hang onto their domains */
    ReductionDomain reduction_domain;

    static Expr make(Type type, const std::string &name) {
        return make(type, name, Buffer<>(), Parameter(), ReductionDomain());
    }

    static Expr make(Type type, const std::string &name, Parameter param) {
        return make(type, name, Buffer<>(), param, ReductionDomain());
    }

    static Expr make(Type type, const std::string &name, Buffer<> image) {
        return make(type, name, image, Parameter(), ReductionDomain());
    }

    static Expr make(Type type, const std::string &name, ReductionDomain reduction_domain) {
        return make(type, name, Buffer<>(), Parameter(), reduction_domain);
    }

    EXPORT static Expr make(Type type, const std::string &name, Buffer<> image,
                            Parameter param, ReductionDomain reduction_domain);

    static const IRNodeType _node_type = IRNodeType::Variable;
};

/** A for loop. Execute the 'body' statement for all values of the
 * variable 'name' from 'min' to 'min + extent'. There are four
 * types of For nodes. A 'Serial' for loop is a conventional
 * one. In a 'Parallel' for loop, each iteration of the loop
 * happens in parallel or in some unspecified order. In a
 * 'Vectorized' for loop, each iteration maps to one SIMD lane,
 * and the whole loop is executed in one shot. For this case,
 * 'extent' must be some small integer constant (probably 4, 8, or
 * 16). An 'Unrolled' for loop compiles to a completely unrolled
 * version of the loop. Each iteration becomes its own
 * statement. Again in this case, 'extent' should be a small
 * integer constant. */
struct For : public StmtNode<For> {
    std::string name;
    Expr min, extent;
    ForType for_type;
    DeviceAPI device_api;
    Stmt body;

    EXPORT static Stmt make(const std::string &name, Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body);

    bool is_parallel() const {
        return (for_type == ForType::Parallel ||
                for_type == ForType::GPUBlock ||
                for_type == ForType::GPUThread);
    }

    static const IRNodeType _node_type = IRNodeType::For;
};

/** Construct a new vector by taking elements from another sequence of
 * vectors. */
struct Shuffle : public ExprNode<Shuffle> {
    std::vector<Expr> vectors;

    /** Indices indicating which vector element to place into the
     * result. The elements are numbered by their position in the
     * concatenation of the vector argumentss. */
    std::vector<int> indices;

    EXPORT static Expr make(const std::vector<Expr> &vectors,
                            const std::vector<int> &indices);

    /** Convenience constructor for making a shuffle representing an
     * interleaving of vectors of the same length. */
    EXPORT static Expr make_interleave(const std::vector<Expr> &vectors);

    /** Convenience constructor for making a shuffle representing a
     * concatenation of the vectors. */
    EXPORT static Expr make_concat(const std::vector<Expr> &vectors);

    /** Convenience constructor for making a shuffle representing a
     * contiguous subset of a vector. */
    EXPORT static Expr make_slice(Expr vector, int begin, int stride, int size);

    /** Convenience constructor for making a shuffle representing
     * extracting a single element. */
    EXPORT static Expr make_extract_element(Expr vector, int i);

    /** Check if this shuffle is an interleaving of the vector
     * arguments. */
    EXPORT bool is_interleave() const;

    /** Check if this shuffle is a concatenation of the vector
     * arguments. */
    EXPORT bool is_concat() const;

    /** Check if this shuffle is a contiguous strict subset of the
     * vector arguments, and if so, the offset and stride of the
     * slice. */
    ///@{
    EXPORT bool is_slice() const;
    int slice_begin() const { return indices[0]; }
    int slice_stride() const { return indices.size() >= 2 ? indices[1] - indices[0] : 1; }
    ///@}

    /** Check if this shuffle is extracting a scalar from the vector
     * arguments. */
    EXPORT bool is_extract_element() const;

    static const IRNodeType _node_type = IRNodeType::Shuffle;
};

/** Represent a multi-dimensional region of a Func or an ImageParam that
 * needs to be prefetched. */
struct Prefetch : public StmtNode<Prefetch> {
    std::string name;
    std::vector<Type> types;
    Region bounds;

    /** If it's a prefetch load from an image parameter, this points to that. */
    Parameter param;

    EXPORT static Stmt make(const std::string &name, const std::vector<Type> &types,
                            const Region &bounds, Parameter param = Parameter());

    static const IRNodeType _node_type = IRNodeType::Prefetch;
};

}
}

#endif
#ifndef HALIDE_BOUNDS_H
#define HALIDE_BOUNDS_H

/** \file
 * Methods for computing the upper and lower bounds of an expression,
 * and the regions of a function read or written by a statement.
 */

#ifndef HALIDE_IR_OPERATOR_H
#define HALIDE_IR_OPERATOR_H

/** \file
 *
 * Defines various operator overloads and utility functions that make
 * it more pleasant to work with Halide expressions.
 */

#include <atomic>


namespace Halide {

namespace Internal {
/** Is the expression either an IntImm, a FloatImm, a StringImm, or a
 * Cast of the same, or a Ramp or Broadcast of the same. Doesn't do
 * any constant folding. */
EXPORT bool is_const(const Expr &e);

/** Is the expression an IntImm, FloatImm of a particular value, or a
 * Cast, or Broadcast of the same. */
EXPORT bool is_const(const Expr &e, int64_t v);

/** If an expression is an IntImm or a Broadcast of an IntImm, return
 * a pointer to its value. Otherwise returns nullptr. */
EXPORT const int64_t *as_const_int(const Expr &e);

/** If an expression is a UIntImm or a Broadcast of a UIntImm, return
 * a pointer to its value. Otherwise returns nullptr. */
EXPORT const uint64_t *as_const_uint(const Expr &e);

/** If an expression is a FloatImm or a Broadcast of a FloatImm,
 * return a pointer to its value. Otherwise returns nullptr. */
EXPORT const double *as_const_float(const Expr &e);

/** Is the expression a constant integer power of two. Also returns
 * log base two of the expression if it is. Only returns true for
 * integer types. */
EXPORT bool is_const_power_of_two_integer(const Expr &e, int *bits);

/** Is the expression a const (as defined by is_const), and also
 * strictly greater than zero (in all lanes, if a vector expression) */
EXPORT bool is_positive_const(const Expr &e);

/** Is the expression a const (as defined by is_const), and also
 * strictly less than zero (in all lanes, if a vector expression) */
EXPORT bool is_negative_const(const Expr &e);

/** Is the expression a const (as defined by is_const), and also
 * strictly less than zero (in all lanes, if a vector expression) and
 * is its negative value representable. (This excludes the most
 * negative value of the Expr's type from inclusion. Intended to be
 * used when the value will be negated as part of simplification.)
 */
EXPORT bool is_negative_negatable_const(const Expr &e);

/** Is the expression an undef */
EXPORT bool is_undef(const Expr &e);

/** Is the expression a const (as defined by is_const), and also equal
 * to zero (in all lanes, if a vector expression) */
EXPORT bool is_zero(const Expr &e);

/** Is the expression a const (as defined by is_const), and also equal
 * to one (in all lanes, if a vector expression) */
EXPORT bool is_one(const Expr &e);

/** Is the expression a const (as defined by is_const), and also equal
 * to two (in all lanes, if a vector expression) */
EXPORT bool is_two(const Expr &e);

/** Is the statement a no-op (which we represent as either an
 * undefined Stmt, or as an Evaluate node of a constant) */
EXPORT bool is_no_op(const Stmt &s);

/** Does the expression
 * 1) Take on the same value no matter where it appears in a Stmt, and
 * 2) Evaluating it has no side-effects
 */
bool is_pure(const Expr &e);

/** Construct an immediate of the given type from any numeric C++ type. */
// @{
EXPORT Expr make_const(Type t, int64_t val);
EXPORT Expr make_const(Type t, uint64_t val);
EXPORT Expr make_const(Type t, double val);
inline Expr make_const(Type t, int32_t val)   {return make_const(t, (int64_t)val);}
inline Expr make_const(Type t, uint32_t val)  {return make_const(t, (uint64_t)val);}
inline Expr make_const(Type t, int16_t val)   {return make_const(t, (int64_t)val);}
inline Expr make_const(Type t, uint16_t val)  {return make_const(t, (uint64_t)val);}
inline Expr make_const(Type t, int8_t val)    {return make_const(t, (int64_t)val);}
inline Expr make_const(Type t, uint8_t val)   {return make_const(t, (uint64_t)val);}
inline Expr make_const(Type t, bool val)      {return make_const(t, (uint64_t)val);}
inline Expr make_const(Type t, float val)     {return make_const(t, (double)val);}
inline Expr make_const(Type t, float16_t val) {return make_const(t, (double)val);}
// @}

/** Check if a constant value can be correctly represented as the given type. */
EXPORT void check_representable(Type t, int64_t val);

/** Construct a boolean constant from a C++ boolean value.
 * May also be a vector if width is given.
 * It is not possible to coerce a C++ boolean to Expr because
 * if we provide such a path then char objects can ambiguously
 * be converted to Halide Expr or to std::string.  The problem
 * is that C++ does not have a real bool type - it is in fact
 * close enough to char that C++ does not know how to distinguish them.
 * make_bool is the explicit coercion. */
EXPORT Expr make_bool(bool val, int lanes = 1);

/** Construct the representation of zero in the given type */
EXPORT Expr make_zero(Type t);

/** Construct the representation of one in the given type */
EXPORT Expr make_one(Type t);

/** Construct the representation of two in the given type */
EXPORT Expr make_two(Type t);

/** Construct the constant boolean true. May also be a vector of
 * trues, if a lanes argument is given. */
EXPORT Expr const_true(int lanes = 1);

/** Construct the constant boolean false. May also be a vector of
 * falses, if a lanes argument is given. */
EXPORT Expr const_false(int lanes = 1);

/** Attempt to cast an expression to a smaller type while provably not
 * losing information. If it can't be done, return an undefined
 * Expr. */
EXPORT Expr lossless_cast(Type t, Expr e);

/** Coerce the two expressions to have the same type, using C-style
 * casting rules. For the purposes of casting, a boolean type is
 * UInt(1). We use the following procedure:
 *
 * If the types already match, do nothing.
 *
 * Then, if one type is a vector and the other is a scalar, the scalar
 * is broadcast to match the vector width, and we continue.
 *
 * Then, if one type is floating-point and the other is not, the
 * non-float is cast to the floating-point type, and we're done.
 *
 * Then, if both types are unsigned ints, the one with fewer bits is
 * cast to match the one with more bits and we're done.
 *
 * Then, if both types are signed ints, the one with fewer bits is
 * cast to match the one with more bits and we're done.
 *
 * Finally, if one type is an unsigned int and the other type is a signed
 * int, both are cast to a signed int with the greater of the two
 * bit-widths. For example, matching an Int(8) with a UInt(16) results
 * in an Int(16).
 *
 */
EXPORT void match_types(Expr &a, Expr &b);

/** Halide's vectorizable transcendentals. */
// @{
EXPORT Expr halide_log(Expr a);
EXPORT Expr halide_exp(Expr a);
EXPORT Expr halide_erf(Expr a);
// @}

/** Raise an expression to an integer power by repeatedly multiplying
 * it by itself. */
EXPORT Expr raise_to_integer_power(Expr a, int64_t b);

/** Split a boolean condition into vector of ANDs. If 'cond' is undefined,
 * return an empty vector. */
EXPORT void split_into_ands(const Expr &cond, std::vector<Expr> &result);

/** A builder to help create Exprs representing halide_buffer_t
 * structs (e.g. foo.buffer) via calls to halide_buffer_init. Fill out
 * the fields and then call build. The resulting Expr will be a call
 * to halide_buffer_init with the struct members as arguments. If the
 * buffer_memory field is undefined, it uses a call to alloca to make
 * some stack memory for the buffer. If the shape_memory field is
 * undefined, it similarly uses stack memory for the shape. If the
 * shape_memory field is null, it uses the dim field already in the
 * buffer. Other unitialized fields will take on a value of zero in
 * the constructed buffer. */
struct BufferBuilder {
    Expr buffer_memory, shape_memory;
    Expr host, device, device_interface;
    Type type;
    int dimensions = 0;
    std::vector<Expr> mins, extents, strides;
    Expr host_dirty, device_dirty;
    EXPORT Expr build() const;
};

/** If e is a ramp expression with stride, default 1, return the base,
 * otherwise undefined. */
Expr strided_ramp_base(Expr e, int stride = 1);

} // namespace Internal

/** Cast an expression to the halide type corresponding to the C++ type T. */
template<typename T>
inline Expr cast(Expr a) {
    return cast(type_of<T>(), std::move(a));
}

/** Cast an expression to a new type. */
inline Expr cast(Type t, Expr a) {
    user_assert(a.defined()) << "cast of undefined Expr\n";
    if (a.type() == t) {
        return a;
    }

    if (t.is_handle() && !a.type().is_handle()) {
        user_error << "Can't cast \"" << a << "\" to a handle. "
                   << "The only legal cast from scalar types to a handle is: "
                   << "reinterpret(Handle(), cast<uint64_t>(" << a << "));\n";
    } else if (a.type().is_handle() && !t.is_handle()) {
        user_error << "Can't cast handle \"" << a << "\" to type " << t << ". "
                   << "The only legal cast from handles to scalar types is: "
                   << "reinterpret(UInt(64), " << a << ");\n";
    }

    // Fold constants early
    if (const int64_t *i = as_const_int(a)) {
        return Internal::make_const(t, *i);
    }
    if (const uint64_t *u = as_const_uint(a)) {
        return Internal::make_const(t, *u);
    }
    if (const double *f = as_const_float(a)) {
        return Internal::make_const(t, *f);
    }

    if (t.is_vector()) {
        if (a.type().is_scalar()) {
            return Internal::Broadcast::make(cast(t.element_of(), std::move(a)), t.lanes());
        } else if (const Internal::Broadcast *b = a.as<Internal::Broadcast>()) {
            internal_assert(b->lanes == t.lanes());
            return Internal::Broadcast::make(cast(t.element_of(), b->value), t.lanes());
        }
    }
    return Internal::Cast::make(t, std::move(a));
}

/** Return the sum of two expressions, doing any necessary type
 * coercion using \ref Internal::match_types */
inline Expr operator+(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator+ of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::Add::make(std::move(a), std::move(b));
}

/** Add an expression and a constant integer. Coerces the type of the
 * integer to match the type of the expression. Errors if the integer
 * cannot be represented in the type of the expression. */
// @{
inline Expr operator+(Expr a, int b) {
    user_assert(a.defined()) << "operator+ of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Add::make(std::move(a), Internal::make_const(t, b));
}

/** Add a constant integer and an expression. Coerces the type of the
 * integer to match the type of the expression. Errors if the integer
 * cannot be represented in the type of the expression. */
inline Expr operator+(int a, Expr b) {
    user_assert(b.defined()) << "operator+ of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Add::make(Internal::make_const(t, a), std::move(b));
}

/** Modify the first expression to be the sum of two expressions,
 * without changing its type. This casts the second argument to match
 * the type of the first. */
inline Expr &operator+=(Expr &a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator+= of undefined Expr\n";
    Type t = a.type();
    a = Internal::Add::make(std::move(a), cast(t, std::move(b)));
    return a;
}

/** Return the difference of two expressions, doing any necessary type
 * coercion using \ref Internal::match_types */
inline Expr operator-(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator- of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::Sub::make(std::move(a), std::move(b));
}

/** Subtracts a constant integer from an expression. Coerces the type of the
 * integer to match the type of the expression. Errors if the integer
 * cannot be represented in the type of the expression. */
inline Expr operator-(Expr a, int b) {
    user_assert(a.defined()) << "operator- of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Sub::make(std::move(a), Internal::make_const(t, b));
}

/** Subtracts an expression from a constant integer. Coerces the type
 * of the integer to match the type of the expression. Errors if the
 * integer cannot be represented in the type of the expression. */
inline Expr operator-(int a, Expr b) {
    user_assert(b.defined()) << "operator- of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Sub::make(Internal::make_const(t, a), std::move(b));
}

/** Return the negative of the argument. Does no type casting, so more
 * formally: return that number which when added to the original,
 * yields zero of the same type. For unsigned integers the negative is
 * still an unsigned integer. E.g. in UInt(8), the negative of 56 is
 * 200, because 56 + 200 == 0 */
inline Expr operator-(Expr a) {
    user_assert(a.defined()) << "operator- of undefined Expr\n";
    Type t = a.type();
    return Internal::Sub::make(Internal::make_zero(t), std::move(a));
}

/** Modify the first expression to be the difference of two expressions,
 * without changing its type. This casts the second argument to match
 * the type of the first. */
inline Expr &operator-=(Expr &a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator-= of undefined Expr\n";
    Type t = a.type();
    a = Internal::Sub::make(std::move(a), cast(t, std::move(b)));
    return a;
}

/** Return the product of two expressions, doing any necessary type
 * coercion using \ref Internal::match_types */
inline Expr operator*(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator* of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::Mul::make(std::move(a), std::move(b));
}

/** Multiply an expression and a constant integer. Coerces the type of the
 * integer to match the type of the expression. Errors if the integer
 * cannot be represented in the type of the expression. */
inline Expr operator*(const Expr &a, int b) {
    user_assert(a.defined()) << "operator* of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Mul::make(std::move(a), Internal::make_const(t, b));
}

/** Multiply a constant integer and an expression. Coerces the type of
 * the integer to match the type of the expression. Errors if the
 * integer cannot be represented in the type of the expression. */
inline Expr operator*(int a, Expr b) {
    user_assert(b.defined()) << "operator* of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Mul::make(Internal::make_const(t, a), std::move(b));
}

/** Modify the first expression to be the product of two expressions,
 * without changing its type. This casts the second argument to match
 * the type of the first. */
inline Expr &operator*=(Expr &a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator*= of undefined Expr\n";
    Type t = a.type();
    a = Internal::Mul::make(std::move(a), cast(t, std::move(b)));
    return a;
}

/** Return the ratio of two expressions, doing any necessary type
 * coercion using \ref Internal::match_types. Note that signed integer
 * division in Halide rounds towards minus infinity, unlike C, which
 * rounds towards zero. */
inline Expr operator/(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator/ of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::Div::make(std::move(a), std::move(b));
}

/** Modify the first expression to be the ratio of two expressions,
 * without changing its type. This casts the second argument to match
 * the type of the first. Note that signed integer division in Halide
 * rounds towards minus infinity, unlike C, which rounds towards
 * zero. */
inline Expr &operator/=(Expr &a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator/= of undefined Expr\n";
    Type t = a.type();
    a = Internal::Div::make(std::move(a), cast(t, std::move(b)));
    return a;
}

/** Divides an expression by a constant integer. Coerces the type
 * of the integer to match the type of the expression. Errors if the
 * integer cannot be represented in the type of the expression. */
inline Expr operator/(Expr a, int b) {
    user_assert(a.defined()) << "operator/ of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Div::make(std::move(a), Internal::make_const(t, b));
}

/** Divides a constant integer by an expression. Coerces the type
 * of the integer to match the type of the expression. Errors if the
 * integer cannot be represented in the type of the expression. */
inline Expr operator/(int a, Expr b) {
    user_assert(b.defined()) << "operator- of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Div::make(Internal::make_const(t, a), std::move(b));
}

/** Return the first argument reduced modulo the second, doing any
 * necessary type coercion using \ref Internal::match_types. For
 * signed integers, the sign of the result matches the sign of the
 * second argument (unlike in C, where it matches the sign of the
 * first argument). For example, this means that x%2 is always either
 * zero or one, even if x is negative.*/
inline Expr operator%(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator% of undefined Expr\n";
    user_assert(!Internal::is_zero(b)) << "operator% with constant 0 modulus\n";
    Internal::match_types(a, b);
    return Internal::Mod::make(std::move(a), std::move(b));
}

/** Mods an expression by a constant integer. Coerces the type
 * of the integer to match the type of the expression. Errors if the
 * integer cannot be represented in the type of the expression. */
inline Expr operator%(Expr a, int b) {
    user_assert(a.defined()) << "operator% of undefined Expr\n";
    user_assert(b != 0) << "operator% with constant 0 modulus\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Mod::make(std::move(a), Internal::make_const(t, b));
}
/** Mods a constant integer by an expression. Coerces the type
 * of the integer to match the type of the expression. Errors if the
 * integer cannot be represented in the type of the expression. */
inline Expr operator%(int a, const Expr &b) {
    user_assert(b.defined()) << "operator% of undefined Expr\n";
    user_assert(!Internal::is_zero(b)) << "operator% with constant 0 modulus\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Mod::make(Internal::make_const(t, a), std::move(b));
}

/** Return a boolean expression that tests whether the first argument
 * is greater than the second, after doing any necessary type coercion
 * using \ref Internal::match_types */
inline Expr operator>(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator> of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::GT::make(std::move(a), std::move(b));
}

/** Return a boolean expression that tests whether an expression is
 * greater than a constant integer. Coerces the integer to the type of
 * the expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator>(Expr a, int b) {
    user_assert(a.defined()) << "operator> of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::GT::make(std::move(a), Internal::make_const(t, b));
}

/** Return a boolean expression that tests whether a constant integer is
 * greater than an expression. Coerces the integer to the type of
 * the expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator>(int a, Expr b) {
    user_assert(b.defined()) << "operator> of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::GT::make(Internal::make_const(t, a), std::move(b));
}

/** Return a boolean expression that tests whether the first argument
 * is less than the second, after doing any necessary type coercion
 * using \ref Internal::match_types */
inline Expr operator<(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator< of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::LT::make(std::move(a), std::move(b));
}

/** Return a boolean expression that tests whether an expression is
 * less than a constant integer. Coerces the integer to the type of
 * the expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator<(Expr a, int b) {
    user_assert(a.defined()) << "operator< of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::LT::make(std::move(a), Internal::make_const(t, b));
}

/** Return a boolean expression that tests whether a constant integer is
 * less than an expression. Coerces the integer to the type of
 * the expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator<(int a, Expr b) {
    user_assert(b.defined()) << "operator< of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::LT::make(Internal::make_const(t, a), std::move(b));
}

/** Return a boolean expression that tests whether the first argument
 * is less than or equal to the second, after doing any necessary type
 * coercion using \ref Internal::match_types */
inline Expr operator<=(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator<= of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::LE::make(std::move(a), std::move(b));
}

/** Return a boolean expression that tests whether an expression is
 * less than or equal to a constant integer. Coerces the integer to
 * the type of the expression. Errors if the integer is not
 * representable in that type. */
inline Expr operator<=(Expr a, int b) {
    user_assert(a.defined()) << "operator<= of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::LE::make(std::move(a), Internal::make_const(t, b));
}

/** Return a boolean expression that tests whether a constant integer
 * is less than or equal to an expression. Coerces the integer to the
 * type of the expression. Errors if the integer is not representable
 * in that type. */
inline Expr operator<=(int a, Expr b) {
    user_assert(b.defined()) << "operator<= of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::LE::make(Internal::make_const(t, a), std::move(b));
}

/** Return a boolean expression that tests whether the first argument
 * is greater than or equal to the second, after doing any necessary
 * type coercion using \ref Internal::match_types */
inline Expr operator>=(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator>= of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::GE::make(std::move(a), std::move(b));
}

/** Return a boolean expression that tests whether an expression is
 * greater than or equal to a constant integer. Coerces the integer to
 * the type of the expression. Errors if the integer is not
 * representable in that type. */
inline Expr operator>=(Expr a, int b) {
    user_assert(a.defined()) << "operator>= of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::GE::make(a, Internal::make_const(t, b));
}

/** Return a boolean expression that tests whether a constant integer
 * is greater than or equal to an expression. Coerces the integer to the
 * type of the expression. Errors if the integer is not representable
 * in that type. */
inline Expr operator>=(int a, Expr b) {
    user_assert(b.defined()) << "operator>= of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::GE::make(Internal::make_const(t, a), b);
}

/** Return a boolean expression that tests whether the first argument
 * is equal to the second, after doing any necessary type coercion
 * using \ref Internal::match_types */
inline Expr operator==(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator== of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::EQ::make(std::move(a), std::move(b));
}

/** Return a boolean expression that tests whether an expression is
 * equal to a constant integer. Coerces the integer to the type of the
 * expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator==(Expr a, int b) {
    user_assert(a.defined()) << "operator== of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::EQ::make(std::move(a), Internal::make_const(t, b));
}

/** Return a boolean expression that tests whether a constant integer
 * is equal to an expression. Coerces the integer to the type of the
 * expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator==(int a, Expr b) {
    user_assert(b.defined()) << "operator== of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::EQ::make(Internal::make_const(t, a), std::move(b));
}

/** Return a boolean expression that tests whether the first argument
 * is not equal to the second, after doing any necessary type coercion
 * using \ref Internal::match_types */
inline Expr operator!=(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "operator!= of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::NE::make(std::move(a), std::move(b));
}

/** Return a boolean expression that tests whether an expression is
 * not equal to a constant integer. Coerces the integer to the type of
 * the expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator!=(Expr a, int b) {
    user_assert(a.defined()) << "operator!= of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::NE::make(std::move(a), Internal::make_const(t, b));
}

/** Return a boolean expression that tests whether a constant integer
 * is not equal to an expression. Coerces the integer to the type of
 * the expression. Errors if the integer is not representable in that
 * type. */
inline Expr operator!=(int a, Expr b) {
    user_assert(b.defined()) << "operator!= of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::NE::make(Internal::make_const(t, a), std::move(b));
}

/** Returns the logical and of the two arguments */
inline Expr operator&&(Expr a, Expr b) {
    Internal::match_types(a, b);
    return Internal::And::make(std::move(a), std::move(b));
}

/** Logical and of an Expr and a bool. Either returns the Expr or an
 * Expr representing false, depending on the bool. */
// @{
inline Expr operator&&(const Expr &a, bool b) {
    internal_assert(a.defined()) << "operator&& of undefined Expr\n";
    internal_assert(a.type().is_bool()) << "operator&& of Expr of type " << a.type() << "\n";
    if (b) {
        return a;
    } else {
        return Internal::make_zero(a.type());
    }
}
inline Expr operator&&(bool a, const Expr &b) {
    return std::move(b) && a;
}
// @}

/** Returns the logical or of the two arguments */
inline Expr operator||(Expr a, Expr b) {
    Internal::match_types(a, b);
    return Internal::Or::make(std::move(a), std::move(b));
}

/** Logical or of an Expr and a bool. Either returns the Expr or an
 * Expr representing true, depending on the bool. */
// @{
inline Expr operator||(const Expr &a, bool b) {
    internal_assert(a.defined()) << "operator|| of undefined Expr\n";
    internal_assert(a.type().is_bool()) << "operator|| of Expr of type " << a.type() << "\n";
    if (b) {
        return Internal::make_one(a.type());
    } else {
        return a;
    }
}
inline Expr operator||(bool a, const Expr &b) {
    return b || a;
}
// @}


/** Returns the logical not the argument */
inline Expr operator!(Expr a) {
    return Internal::Not::make(std::move(a));
}

/** Returns an expression representing the greater of the two
 * arguments, after doing any necessary type coercion using
 * \ref Internal::match_types. Vectorizes cleanly on most platforms
 * (with the exception of integer types on x86 without SSE4). */
inline Expr max(Expr a, Expr b) {
    user_assert(a.defined() && b.defined())
        << "max of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::Max::make(std::move(a), std::move(b));
}

/** Returns an expression representing the greater of an expression
 * and a constant integer.  The integer is coerced to the type of the
 * expression. Errors if the integer is not representable as that
 * type. Vectorizes cleanly on most platforms (with the exception of
 * integer types on x86 without SSE4). */
inline Expr max(Expr a, int b) {
    user_assert(a.defined()) << "max of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Max::make(std::move(a), Internal::make_const(t, b));
}


/** Returns an expression representing the greater of a constant
 * integer and an expression. The integer is coerced to the type of
 * the expression. Errors if the integer is not representable as that
 * type. Vectorizes cleanly on most platforms (with the exception of
 * integer types on x86 without SSE4). */
inline Expr max(int a, Expr b) {
    user_assert(b.defined()) << "max of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Max::make(Internal::make_const(t, a), std::move(b));
}

inline Expr max(float a, Expr b) {return max(Expr(a), std::move(b));}
inline Expr max(Expr a, float b) {return max(std::move(a), Expr(b));}

/** Returns an expression representing the greater of an expressions
 * vector, after doing any necessary type coersion using
 * \ref Internal::match_types. Vectorizes cleanly on most platforms
 * (with the exception of integer types on x86 without SSE4).
 * The expressions are folded from right ie. max(.., max(.., ..)).
 * The arguments can be any mix of types but must all be convertible to Expr. */
template<typename A, typename B, typename C, typename... Rest,
         typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Rest...>::value>::type* = nullptr>
inline Expr max(A &&a, B &&b, C &&c, Rest&&... rest) {
    return max(std::forward<A>(a), max(std::forward<B>(b), std::forward<C>(c), std::forward<Rest>(rest)...));
}

inline Expr min(Expr a, Expr b) {
    user_assert(a.defined() && b.defined())
        << "min of undefined Expr\n";
    Internal::match_types(a, b);
    return Internal::Min::make(std::move(a), std::move(b));
}

/** Returns an expression representing the lesser of an expression
 * and a constant integer.  The integer is coerced to the type of the
 * expression. Errors if the integer is not representable as that
 * type. Vectorizes cleanly on most platforms (with the exception of
 * integer types on x86 without SSE4). */
inline Expr min(Expr a, int b) {
    user_assert(a.defined()) << "max of undefined Expr\n";
    Type t = a.type();
    Internal::check_representable(t, b);
    return Internal::Min::make(std::move(a), Internal::make_const(t, b));
}

/** Returns an expression representing the lesser of a constant
 * integer and an expression. The integer is coerced to the type of
 * the expression. Errors if the integer is not representable as that
 * type. Vectorizes cleanly on most platforms (with the exception of
 * integer types on x86 without SSE4). */
inline Expr min(int a, Expr b) {
    user_assert(b.defined()) << "max of undefined Expr\n";
    Type t = b.type();
    Internal::check_representable(t, a);
    return Internal::Min::make(Internal::make_const(t, a), std::move(b));
}

inline Expr min(float a, Expr b) {return min(Expr(a), std::move(b));}
inline Expr min(Expr a, float b) {return min(std::move(a), Expr(b));}

/** Returns an expression representing the lesser of an expressions
 * vector, after doing any necessary type coersion using
 * \ref Internal::match_types. Vectorizes cleanly on most platforms
 * (with the exception of integer types on x86 without SSE4).
 * The expressions are folded from right ie. min(.., min(.., ..)).
 * The arguments can be any mix of types but must all be convertible to Expr. */
template<typename A, typename B, typename C, typename... Rest,
         typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Rest...>::value>::type* = nullptr>
inline Expr min(A &&a, B &&b, C &&c, Rest&&... rest) {
    return min(std::forward<A>(a), min(std::forward<B>(b), std::forward<C>(c), std::forward<Rest>(rest)...));
}

/** Operators on floats treats those floats as Exprs. Making these
 * explicit prevents implicit float->int casts that might otherwise
 * occur. */
// @{
inline Expr operator+(Expr a, float b) {return std::move(a) + Expr(b);}
inline Expr operator+(float a, Expr b) {return Expr(a) + std::move(b);}
inline Expr operator-(Expr a, float b) {return std::move(a) - Expr(b);}
inline Expr operator-(float a, Expr b) {return Expr(a) - std::move(b);}
inline Expr operator*(Expr a, float b) {return std::move(a) * Expr(b);}
inline Expr operator*(float a, Expr b) {return Expr(a) * std::move(b);}
inline Expr operator/(Expr a, float b) {return std::move(a) / Expr(b);}
inline Expr operator/(float a, Expr b) {return Expr(a) / std::move(b);}
inline Expr operator%(Expr a, float b) {return std::move(a) % Expr(b);}
inline Expr operator%(float a, Expr b) {return Expr(a) % std::move(b);}
inline Expr operator>(Expr a, float b) {return std::move(a) > Expr(b);}
inline Expr operator>(float a, Expr b) {return Expr(a) > std::move(b);}
inline Expr operator<(Expr a, float b) {return std::move(a) < Expr(b);}
inline Expr operator<(float a, Expr b) {return Expr(a) < std::move(b);}
inline Expr operator>=(Expr a, float b) {return std::move(a) >= Expr(b);}
inline Expr operator>=(float a, Expr b) {return Expr(a) >= std::move(b);}
inline Expr operator<=(Expr a, float b) {return std::move(a) <= Expr(b);}
inline Expr operator<=(float a, Expr b) {return Expr(a) <= std::move(b);}
inline Expr operator==(Expr a, float b) {return std::move(a) == Expr(b);}
inline Expr operator==(float a, Expr b) {return Expr(a) == std::move(b);}
inline Expr operator!=(Expr a, float b) {return std::move(a) != Expr(b);}
inline Expr operator!=(float a, Expr b) {return Expr(a) != std::move(b);}
// @}

/** Clamps an expression to lie within the given bounds. The bounds
 * are type-cast to match the expression. Vectorizes as well as min/max. */
inline Expr clamp(Expr a, Expr min_val, Expr max_val) {
    user_assert(a.defined() && min_val.defined() && max_val.defined())
        << "clamp of undefined Expr\n";
    Expr n_min_val = lossless_cast(a.type(), min_val);
    user_assert(n_min_val.defined())
        << "Type mismatch in call to clamp. First argument ("
        << a << ") has type " << a.type() << ", but second argument ("
        << min_val << ") has type " << min_val.type() << ". Use an explicit cast.\n";
    Expr n_max_val = lossless_cast(a.type(), max_val);
    user_assert(n_max_val.defined())
        << "Type mismatch in call to clamp. First argument ("
        << a << ") has type " << a.type() << ", but third argument ("
        << max_val << ") has type " << max_val.type() << ". Use an explicit cast.\n";
    return Internal::Max::make(Internal::Min::make(std::move(a), std::move(n_max_val)), std::move(n_min_val));
}

/** Returns the absolute value of a signed integer or floating-point
 * expression. Vectorizes cleanly. Unlike in C, abs of a signed
 * integer returns an unsigned integer of the same bit width. This
 * means that abs of the most negative integer doesn't overflow. */
inline Expr abs(Expr a) {
    user_assert(a.defined())
        << "abs of undefined Expr\n";
    Type t = a.type();
    if (t.is_uint()) {
        user_warning << "Warning: abs of an unsigned type is a no-op\n";
        return a;
    }
    return Internal::Call::make(t.with_code(t.is_int() ? Type::UInt : t.code()),
                                Internal::Call::abs, {std::move(a)}, Internal::Call::PureIntrinsic);
}

/** Return the absolute difference between two values. Vectorizes
 * cleanly. Returns an unsigned value of the same bit width. There are
 * various ways to write this yourself, but they contain numerous
 * gotchas and don't always compile to good code, so use this
 * instead. */
inline Expr absd(Expr a, Expr b) {
    user_assert(a.defined() && b.defined()) << "absd of undefined Expr\n";
    Internal::match_types(a, b);
    Type t = a.type();

    if (t.is_float()) {
        // Floats can just use abs.
        return abs(std::move(a) - std::move(b));
    }

    // The argument may be signed, but the return type is unsigned.
    return Internal::Call::make(t.with_code(t.is_int() ? Type::UInt : t.code()),
                                Internal::Call::absd, {std::move(a), std::move(b)},
                                Internal::Call::PureIntrinsic);
}

/** Returns an expression similar to the ternary operator in C, except
 * that it always evaluates all arguments. If the first argument is
 * true, then return the second, else return the third. Typically
 * vectorizes cleanly, but benefits from SSE41 or newer on x86. */
inline Expr select(Expr condition, Expr true_value, Expr false_value) {

    if (as_const_int(condition)) {
        // Why are you doing this? We'll preserve the select node until constant folding for you.
        condition = cast(Bool(), std::move(condition));
    }

    // Coerce int literals to the type of the other argument
    if (as_const_int(true_value)) {
        true_value = cast(false_value.type(), std::move(true_value));
    }
    if (as_const_int(false_value)) {
        false_value = cast(true_value.type(), std::move(false_value));
    }

    user_assert(condition.type().is_bool())
        << "The first argument to a select must be a boolean:\n"
        << "  " << condition << " has type " << condition.type() << "\n";

    user_assert(true_value.type() == false_value.type())
        << "The second and third arguments to a select do not have a matching type:\n"
        << "  " << true_value << " has type " << true_value.type() << "\n"
        << "  " << false_value << " has type " << false_value.type() << "\n";

    return Internal::Select::make(std::move(condition), std::move(true_value), std::move(false_value));
}

/** A multi-way variant of select similar to a switch statement in C,
 * which can accept multiple conditions and values in pairs. Evaluates
 * to the first value for which the condition is true. Returns the
 * final value if all conditions are false. */
template<typename... Args,
         typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Args...>::value>::type* = nullptr>
inline Expr select(Expr c0, Expr v0, Expr c1, Expr v1, Args&&... args) {
    return select(std::move(c0), std::move(v0), select(std::move(c1), std::move(v1), std::forward<Args>(args)...));
}

// TODO: Implement support for *_f16 external functions in various backends.
// No backend supports these yet.

/** Return the sine of a floating-point expression. If the argument is
 * not floating-point, it is cast to Float(32). Does not vectorize
 * well. */
inline Expr sin(Expr x) {
    user_assert(x.defined()) << "sin of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "sin_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "sin_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "sin_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the arcsine of a floating-point expression. If the argument
 * is not floating-point, it is cast to Float(32). Does not vectorize
 * well. */
inline Expr asin(Expr x) {
    user_assert(x.defined()) << "asin of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "asin_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "asin_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "asin_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the cosine of a floating-point expression. If the argument
 * is not floating-point, it is cast to Float(32). Does not vectorize
 * well. */
inline Expr cos(Expr x) {
    user_assert(x.defined()) << "cos of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "cos_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "cos_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "cos_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the arccosine of a floating-point expression. If the
 * argument is not floating-point, it is cast to Float(32). Does not
 * vectorize well. */
inline Expr acos(Expr x) {
    user_assert(x.defined()) << "acos of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "acos_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "acos_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "acos_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the tangent of a floating-point expression. If the argument
 * is not floating-point, it is cast to Float(32). Does not vectorize
 * well. */
inline Expr tan(Expr x) {
    user_assert(x.defined()) << "tan of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "tan_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "tan_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "tan_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the arctangent of a floating-point expression. If the
 * argument is not floating-point, it is cast to Float(32). Does not
 * vectorize well. */
inline Expr atan(Expr x) {
    user_assert(x.defined()) << "atan of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "atan_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "atan_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "atan_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the angle of a floating-point gradient. If the argument is
 * not floating-point, it is cast to Float(32). Does not vectorize
 * well. */
inline Expr atan2(Expr y, Expr x) {
    user_assert(x.defined() && y.defined()) << "atan2 of undefined Expr\n";

    if (y.type() == Float(64)) {
        x = cast<double>(x);
        return Internal::Call::make(Float(64), "atan2_f64", {std::move(y), std::move(x)}, Internal::Call::PureExtern);
    } else if (y.type() == Float(16)) {
        x = cast<float16_t>(x);
        return Internal::Call::make(Float(16), "atan2_f16", {std::move(y), std::move(x)}, Internal::Call::PureExtern);
    } else {
        y = cast<float>(y);
        x = cast<float>(x);
        return Internal::Call::make(Float(32), "atan2_f32", {std::move(y), std::move(x)}, Internal::Call::PureExtern);
    }
}

/** Return the hyperbolic sine of a floating-point expression.  If the
 *  argument is not floating-point, it is cast to Float(32). Does not
 *  vectorize well. */
inline Expr sinh(Expr x) {
    user_assert(x.defined()) << "sinh of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "sinh_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "sinh_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "sinh_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the hyperbolic arcsinhe of a floating-point expression.  If
 * the argument is not floating-point, it is cast to Float(32). Does
 * not vectorize well. */
inline Expr asinh(Expr x) {
    user_assert(x.defined()) << "asinh of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "asinh_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "asinh_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "asinh_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the hyperbolic cosine of a floating-point expression.  If
 * the argument is not floating-point, it is cast to Float(32). Does
 * not vectorize well. */
inline Expr cosh(Expr x) {
    user_assert(x.defined()) << "cosh of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "cosh_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "cosh_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "cosh_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the hyperbolic arccosine of a floating-point expression.
 * If the argument is not floating-point, it is cast to
 * Float(32). Does not vectorize well. */
inline Expr acosh(Expr x) {
    user_assert(x.defined()) << "acosh of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "acosh_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "acosh_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "acosh_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the hyperbolic tangent of a floating-point expression.  If
 * the argument is not floating-point, it is cast to Float(32). Does
 * not vectorize well. */
inline Expr tanh(Expr x) {
    user_assert(x.defined()) << "tanh of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "tanh_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "tanh_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "tanh_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the hyperbolic arctangent of a floating-point expression.
 * If the argument is not floating-point, it is cast to
 * Float(32). Does not vectorize well. */
inline Expr atanh(Expr x) {
    user_assert(x.defined()) << "atanh of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "atanh_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "atanh_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "atanh_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the square root of a floating-point expression. If the
 * argument is not floating-point, it is cast to Float(32). Typically
 * vectorizes cleanly. */
inline Expr sqrt(Expr x) {
    user_assert(x.defined()) << "sqrt of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "sqrt_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "sqrt_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "sqrt_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the square root of the sum of the squares of two
 * floating-point expressions. If the argument is not floating-point,
 * it is cast to Float(32). Vectorizes cleanly. */
inline Expr hypot(Expr x, Expr y) {
    return sqrt(x * x + y * y);
}

/** Return the exponential of a floating-point expression. If the
 * argument is not floating-point, it is cast to Float(32). For
 * Float(64) arguments, this calls the system exp function, and does
 * not vectorize well. For Float(32) arguments, this function is
 * vectorizable, does the right thing for extremely small or extremely
 * large inputs, and is accurate up to the last bit of the
 * mantissa. Vectorizes cleanly. */
inline Expr exp(Expr x) {
    user_assert(x.defined()) << "exp of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "exp_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "exp_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "exp_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the logarithm of a floating-point expression. If the
 * argument is not floating-point, it is cast to Float(32). For
 * Float(64) arguments, this calls the system log function, and does
 * not vectorize well. For Float(32) arguments, this function is
 * vectorizable, does the right thing for inputs <= 0 (returns -inf or
 * nan), and is accurate up to the last bit of the
 * mantissa. Vectorizes cleanly. */
inline Expr log(Expr x) {
    user_assert(x.defined()) << "log of undefined Expr\n";
    if (x.type() == Float(64)) {
        return Internal::Call::make(Float(64), "log_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        return Internal::Call::make(Float(16), "log_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        return Internal::Call::make(Float(32), "log_f32", {cast<float>(std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return one floating point expression raised to the power of
 * another. The type of the result is given by the type of the first
 * argument. If the first argument is not a floating-point type, it is
 * cast to Float(32). For Float(32), cleanly vectorizable, and
 * accurate up to the last few bits of the mantissa. Gets worse when
 * approaching overflow. Vectorizes cleanly. */
inline Expr pow(Expr x, Expr y) {
    user_assert(x.defined() && y.defined()) << "pow of undefined Expr\n";

    if (const int64_t *i = as_const_int(y)) {
        return raise_to_integer_power(std::move(x), *i);
    }

    if (x.type() == Float(64)) {
        y = cast<double>(std::move(y));
        return Internal::Call::make(Float(64), "pow_f64", {std::move(x), std::move(y)}, Internal::Call::PureExtern);
    } else if (x.type() == Float(16)) {
        y = cast<float16_t>(std::move(y));
        return Internal::Call::make(Float(16), "pow_f16", {std::move(x), std::move(y)}, Internal::Call::PureExtern);
    } else {
        x = cast<float>(std::move(x));
        y = cast<float>(std::move(y));
        return Internal::Call::make(Float(32), "pow_f32", {std::move(x), std::move(y)}, Internal::Call::PureExtern);
    }
}

/** Evaluate the error function erf. Only available for
 * Float(32). Accurate up to the last three bits of the
 * mantissa. Vectorizes cleanly. */
inline Expr erf(Expr x) {
    user_assert(x.defined()) << "erf of undefined Expr\n";
    user_assert(x.type() == Float(32)) << "erf only takes float arguments\n";
    return Internal::halide_erf(std::move(x));
}

/** Fast approximate cleanly vectorizable log for Float(32). Returns
 * nonsense for x <= 0.0f. Accurate up to the last 5 bits of the
 * mantissa. Vectorizes cleanly. */
EXPORT Expr fast_log(Expr x);

/** Fast approximate cleanly vectorizable exp for Float(32). Returns
 * nonsense for inputs that would overflow or underflow. Typically
 * accurate up to the last 5 bits of the mantissa. Gets worse when
 * approaching overflow. Vectorizes cleanly. */
EXPORT Expr fast_exp(Expr x);

/** Fast approximate cleanly vectorizable pow for Float(32). Returns
 * nonsense for x < 0.0f. Accurate up to the last 5 bits of the
 * mantissa for typical exponents. Gets worse when approaching
 * overflow. Vectorizes cleanly. */
inline Expr fast_pow(Expr x, Expr y) {
    if (const int64_t *i = as_const_int(y)) {
        return raise_to_integer_power(std::move(x), *i);
    }

    x = cast<float>(std::move(x));
    y = cast<float>(std::move(y));
    return select(x == 0.0f, 0.0f, fast_exp(fast_log(x) * std::move(y)));
}

/** Fast approximate inverse for Float(32). Corresponds to the rcpps
 * instruction on x86, and the vrecpe instruction on ARM. Vectorizes
 * cleanly. */
inline Expr fast_inverse(Expr x) {
    user_assert(x.type() == Float(32)) << "fast_inverse only takes float arguments\n";
    Type t = x.type();
    return Internal::Call::make(t, "fast_inverse_f32", {std::move(x)}, Internal::Call::PureExtern);
}

/** Fast approximate inverse square root for Float(32). Corresponds to
 * the rsqrtps instruction on x86, and the vrsqrte instruction on
 * ARM. Vectorizes cleanly. */
inline Expr fast_inverse_sqrt(Expr x) {
    user_assert(x.type() == Float(32)) << "fast_inverse_sqrt only takes float arguments\n";
    Type t = x.type();
    return Internal::Call::make(t, "fast_inverse_sqrt_f32", {std::move(x)}, Internal::Call::PureExtern);
}

/** Return the greatest whole number less than or equal to a
 * floating-point expression. If the argument is not floating-point,
 * it is cast to Float(32). The return value is still in floating
 * point, despite being a whole number. Vectorizes cleanly. */
inline Expr floor(Expr x) {
    user_assert(x.defined()) << "floor of undefined Expr\n";
    Type t = x.type();
    if (t.element_of() == Float(64)) {
        return Internal::Call::make(t, "floor_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (t.element_of() == Float(16)) {
        return Internal::Call::make(t, "floor_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        t = t.with_code(Type::Float);
        return Internal::Call::make(t, "floor_f32", {cast(t, std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the least whole number greater than or equal to a
 * floating-point expression. If the argument is not floating-point,
 * it is cast to Float(32). The return value is still in floating
 * point, despite being a whole number. Vectorizes cleanly. */
inline Expr ceil(Expr x) {
    user_assert(x.defined()) << "ceil of undefined Expr\n";
    Type t = x.type();
    if (t.element_of() == Float(64)) {
        return Internal::Call::make(t, "ceil_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type().element_of() == Float(16)) {
        return Internal::Call::make(t, "ceil_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        t = t.with_code(Type::Float);
        return Internal::Call::make(t, "ceil_f32", {cast(t, std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the whole number closest to a floating-point expression. If the
 * argument is not floating-point, it is cast to Float(32). The return value
 * is still in floating point, despite being a whole number. On ties, we
 * follow IEEE754 conventions and round to the nearest even number. Vectorizes
 * cleanly. */
inline Expr round(Expr x) {
    user_assert(x.defined()) << "round of undefined Expr\n";
    Type t = x.type();
    if (t.element_of() == Float(64)) {
        return Internal::Call::make(t, "round_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (t.element_of() == Float(16)) {
        return Internal::Call::make(t, "round_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        t = t.with_code(Type::Float);
        return Internal::Call::make(t, "round_f32", {cast(t, std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the integer part of a floating-point expression. If the argument is
 * not floating-point, it is cast to Float(32). The return value is still in
 * floating point, despite being a whole number. Vectorizes cleanly. */
inline Expr trunc(Expr x) {
    user_assert(x.defined()) << "trunc of undefined Expr\n";
    Type t = x.type();
    if (t.element_of() == Float(64)) {
        return Internal::Call::make(t, "trunc_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (t.element_of() == Float(16)) {
        return Internal::Call::make(t, "trunc_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        t = t.with_code(Type::Float);
        return Internal::Call::make(t, "trunc_f32", {cast(t, std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Returns true if the argument is a Not a Number (NaN). Requires a
  * floating point argument.  Vectorizes cleanly. */
inline Expr is_nan(Expr x) {
    user_assert(x.defined()) << "is_nan of undefined Expr\n";
    user_assert(x.type().is_float()) << "is_nan only works for float";
    Type t = Bool(x.type().lanes());
    if (x.type().element_of() == Float(64)) {
        return Internal::Call::make(t, "is_nan_f64", {std::move(x)}, Internal::Call::PureExtern);
    } else if (x.type().element_of() == Float(64)) {
        return Internal::Call::make(t, "is_nan_f16", {std::move(x)}, Internal::Call::PureExtern);
    } else {
        Type ft = x.type().with_code(Type::Float);
        return Internal::Call::make(t, "is_nan_f32", {cast(ft, std::move(x))}, Internal::Call::PureExtern);
    }
}

/** Return the fractional part of a floating-point expression. If the argument
 *  is not floating-point, it is cast to Float(32). The return value has the
 *  same sign as the original expression. Vectorizes cleanly. */
inline Expr fract(Expr x) {
    user_assert(x.defined()) << "fract of undefined Expr\n";
    return x - trunc(x);
}

/** Reinterpret the bits of one value as another type. */
inline Expr reinterpret(Type t, Expr e) {
    user_assert(e.defined()) << "reinterpret of undefined Expr\n";
    int from_bits = e.type().bits() * e.type().lanes();
    int to_bits = t.bits() * t.lanes();
    user_assert(from_bits == to_bits)
        << "Reinterpret cast from type " << e.type()
        << " which has " << from_bits
        << " bits, to type " << t
        << " which has " << to_bits << " bits\n";
    return Internal::Call::make(t, Internal::Call::reinterpret, {std::move(e)}, Internal::Call::PureIntrinsic);
}

template<typename T>
inline Expr reinterpret(Expr e) {
    return reinterpret(type_of<T>(), e);
}

/** Return the bitwise and of two expressions (which need not have the
 * same type). The type of the result is the type of the first
 * argument. */
inline Expr operator&(Expr x, Expr y) {
    user_assert(x.defined() && y.defined()) << "bitwise and of undefined Expr\n";
    user_assert(x.type().is_int() || x.type().is_uint())
        << "The first argument to bitwise and must be an integer or unsigned integer";
    user_assert(y.type().is_int() || y.type().is_uint())
        << "The second argument to bitwise and must be an integer or unsigned integer";
    // First widen or narrow, then bitcast.
    if (y.type().bits() != x.type().bits()) {
        y = cast(y.type().with_bits(x.type().bits()), y);
    }
    if (y.type() != x.type()) {
        y = reinterpret(x.type(), y);
    }
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::bitwise_and, {std::move(x), std::move(y)}, Internal::Call::PureIntrinsic);
}

/** Return the bitwise or of two expressions (which need not have the
 * same type). The type of the result is the type of the first
 * argument. */
inline Expr operator|(Expr x, Expr y) {
    user_assert(x.defined() && y.defined()) << "bitwise or of undefined Expr\n";
    user_assert(x.type().is_int() || x.type().is_uint())
        << "The first argument to bitwise or must be an integer or unsigned integer";
    user_assert(y.type().is_int() || y.type().is_uint())
        << "The second argument to bitwise or must be an integer or unsigned integer";
    // First widen or narrow, then bitcast.
    if (y.type().bits() != x.type().bits()) {
        y = cast(y.type().with_bits(x.type().bits()), y);
    }
    if (y.type() != x.type()) {
        y = reinterpret(x.type(), y);
    }
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::bitwise_or, {std::move(x), std::move(y)}, Internal::Call::PureIntrinsic);
}

/** Return the bitwise exclusive or of two expressions (which need not
 * have the same type). The type of the result is the type of the
 * first argument. */
inline Expr operator^(Expr x, Expr y) {
    user_assert(x.defined() && y.defined()) << "bitwise xor of undefined Expr\n";
    user_assert(x.type().is_int() || x.type().is_uint())
        << "The first argument to bitwise xor must be an integer or unsigned integer";
    user_assert(y.type().is_int() || y.type().is_uint())
        << "The second argument to bitwise xor must be an integer or unsigned integer";
    // First widen or narrow, then bitcast.
    if (y.type().bits() != x.type().bits()) {
        y = cast(y.type().with_bits(x.type().bits()), y);
    }
    if (y.type() != x.type()) {
        y = reinterpret(x.type(), y);
    }
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::bitwise_xor, {std::move(x), std::move(y)}, Internal::Call::PureIntrinsic);
}

/** Return the bitwise not of an expression. */
inline Expr operator~(Expr x) {
    user_assert(x.defined()) << "bitwise not of undefined Expr\n";
    user_assert(x.type().is_int() || x.type().is_uint())
        << "Argument to bitwise not must be an integer or unsigned integer";
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::bitwise_not, {std::move(x)}, Internal::Call::PureIntrinsic);
}

/** Shift the bits of an integer value left. This is actually less
 * efficient than multiplying by 2^n, because Halide's optimization
 * passes understand multiplication, and will compile it to
 * shifting. This operator is only for if you really really need bit
 * shifting (e.g. because the exponent is a run-time parameter). The
 * type of the result is equal to the type of the first argument. Both
 * arguments must have integer type. */
// @{
inline Expr operator<<(Expr x, Expr y) {
    user_assert(x.defined() && y.defined()) << "shift left of undefined Expr\n";
    user_assert(!x.type().is_float()) << "First argument to shift left is a float: " << x << "\n";
    user_assert(!y.type().is_float()) << "Second argument to shift left is a float: " << y << "\n";
    Internal::match_types(x, y);
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::shift_left, {std::move(x), std::move(y)}, Internal::Call::PureIntrinsic);
}
inline Expr operator<<(Expr x, int y) {
    Type t = x.type();
    Internal::check_representable(t, y);
    return std::move(x) << Internal::make_const(t, y);
}
inline Expr operator<<(int x, Expr y) {
    Type t = y.type();
    Internal::check_representable(t, x);
    return Internal::make_const(t, x) << std::move(y);
}
// @}

/** Shift the bits of an integer value right. Does sign extension for
 * signed integers. This is less efficient than dividing by a power of
 * two. Halide's definition of division (always round to negative
 * infinity) means that all divisions by powers of two get compiled to
 * bit-shifting, and Halide's optimization routines understand
 * division and can work with it. The type of the result is equal to
 * the type of the first argument. Both arguments must have integer
 * type. */
// @{
inline Expr operator>>(Expr x, Expr y) {
    user_assert(x.defined() && y.defined()) << "shift right of undefined Expr\n";
    user_assert(!x.type().is_float()) << "First argument to shift right is a float: " << x << "\n";
    user_assert(!y.type().is_float()) << "Second argument to shift right is a float: " << y << "\n";
    Internal::match_types(x, y);
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::shift_right, {std::move(x), std::move(y)}, Internal::Call::PureIntrinsic);
}
inline Expr operator>>(Expr x, int y) {
    Type t = x.type();
    Internal::check_representable(t, y);
    return std::move(x) >> Internal::make_const(t, y);
}
inline Expr operator>>(int x, Expr y) {
    Type t = y.type();
    Internal::check_representable(t, x);
    return Internal::make_const(t, x) >> std::move(y);
}
// @}

/** Linear interpolate between the two values according to a weight.
 * \param zero_val The result when weight is 0
 * \param one_val The result when weight is 1
 * \param weight The interpolation amount
 *
 * Both zero_val and one_val must have the same type. All types are
 * supported, including bool.
 *
 * The weight is treated as its own type and must be float or an
 * unsigned integer type. It is scaled to the bit-size of the type of
 * x and y if they are integer, or converted to float if they are
 * float. Integer weights are converted to float via division by the
 * full-range value of the weight's type. Floating-point weights used
 * to interpolate between integer values must be between 0.0f and
 * 1.0f, and an error may be signaled if it is not provably so. (clamp
 * operators can be added to provide proof. Currently an error is only
 * signalled for constant weights.)
 *
 * For integer linear interpolation, out of range values cannot be
 * represented. In particular, weights that are conceptually less than
 * 0 or greater than 1.0 are not representable. As such the result is
 * always between x and y (inclusive of course). For lerp with
 * floating-point values and floating-point weight, the full range of
 * a float is valid, however underflow and overflow can still occur.
 *
 * Ordering is not required between zero_val and one_val:
 *     lerp(42, 69, .5f) == lerp(69, 42, .5f) == 56
 *
 * Results for integer types are for exactly rounded arithmetic. As
 * such, there are cases where 16-bit and float differ because 32-bit
 * floating-point (float) does not have enough precision to produce
 * the exact result. (Likely true for 32-bit integer
 * vs. double-precision floating-point as well.)
 *
 * At present, double precision and 64-bit integers are not supported.
 *
 * Generally, lerp will vectorize as if it were an operation on a type
 * twice the bit size of the inferred type for x and y.
 *
 * Some examples:
 * \code
 *
 *     // Since Halide does not have direct type delcarations, casts
 *     // below are used to indicate the types of the parameters.
 *     // Such casts not required or expected in actual code where types
 *     // are inferred.
 *
 *     lerp(cast<float>(x), cast<float>(y), cast<float>(w)) ->
 *       x * (1.0f - w) + y * w
 *
 *     lerp(cast<uint8_t>(x), cast<uint8_t>(y), cast<uint8_t>(w)) ->
 *       cast<uint8_t>(cast<uint8_t>(x) * (1.0f - cast<uint8_t>(w) / 255.0f) +
 *                     cast<uint8_t>(y) * cast<uint8_t>(w) / 255.0f + .5f)
 *
 *     // Note addition in Halide promoted uint8_t + int8_t to int16_t already,
 *     // the outer cast is added for clarity.
 *     lerp(cast<uint8_t>(x), cast<int8_t>(y), cast<uint8_t>(w)) ->
 *       cast<int16_t>(cast<uint8_t>(x) * (1.0f - cast<uint8_t>(w) / 255.0f) +
 *                     cast<int8_t>(y) * cast<uint8_t>(w) / 255.0f + .5f)
 *
 *     lerp(cast<int8_t>(x), cast<int8_t>(y), cast<float>(w)) ->
 *       cast<int8_t>(cast<int8_t>(x) * (1.0f - cast<float>(w)) +
 *                    cast<int8_t>(y) * cast<uint8_t>(w))
 *
 * \endcode
 * */
inline Expr lerp(Expr zero_val, Expr one_val, Expr weight) {
    user_assert(zero_val.defined()) << "lerp with undefined zero value";
    user_assert(one_val.defined()) << "lerp with undefined one value";
    user_assert(weight.defined()) << "lerp with undefined weight";

    // We allow integer constants through, so that you can say things
    // like lerp(0, cast<uint8_t>(x), alpha) and produce an 8-bit
    // result. Note that lerp(0.0f, cast<uint8_t>(x), alpha) will
    // produce an error, as will lerp(0.0f, cast<double>(x),
    // alpha). lerp(0, cast<float>(x), alpha) is also allowed and will
    // produce a float result.
    if (as_const_int(zero_val)) {
        zero_val = cast(one_val.type(), std::move(zero_val));
    }
    if (as_const_int(one_val)) {
        one_val = cast(zero_val.type(), std::move(one_val));
    }

    user_assert(zero_val.type() == one_val.type())
        << "Can't lerp between " << zero_val << " of type " << zero_val.type()
        << " and " << one_val << " of different type " << one_val.type() << "\n";
    user_assert((weight.type().is_uint() || weight.type().is_float()))
        << "A lerp weight must be an unsigned integer or a float, but "
        << "lerp weight " << weight << " has type " << weight.type() << ".\n";
    user_assert((zero_val.type().is_float() || zero_val.type().lanes() <= 32))
        << "Lerping between 64-bit integers is not supported\n";
    // Compilation error for constant weight that is out of range for integer use
    // as this seems like an easy to catch gotcha.
    if (!zero_val.type().is_float()) {
        const double *const_weight = as_const_float(weight);
        if (const_weight) {
            user_assert(*const_weight >= 0.0 && *const_weight <= 1.0)
                << "Floating-point weight for lerp with integer arguments is "
                << *const_weight << ", which is not in the range [0.0, 1.0].\n";
        }
    }
    Type t = zero_val.type();
    return Internal::Call::make(t, Internal::Call::lerp,
                                {std::move(zero_val), std::move(one_val), std::move(weight)},
                                Internal::Call::PureIntrinsic);
}

/** Count the number of set bits in an expression. */
inline Expr popcount(Expr x) {
    user_assert(x.defined()) << "popcount of undefined Expr\n";
    Type t = x.type();
    user_assert(t.is_uint() || t.is_int())
        << "Argument to popcount must be an integer\n";
    return Internal::Call::make(t, Internal::Call::popcount,
                                {std::move(x)}, Internal::Call::PureIntrinsic);
}

/** Count the number of leading zero bits in an expression. The result is
 *  undefined if the value of the expression is zero. */
inline Expr count_leading_zeros(Expr x) {
    user_assert(x.defined()) << "count leading zeros of undefined Expr\n";
    Type t = x.type();
    user_assert(t.is_uint() || t.is_int())
        << "Argument to count_leading_zeros must be an integer\n";
    return Internal::Call::make(t, Internal::Call::count_leading_zeros,
                                {std::move(x)}, Internal::Call::PureIntrinsic);
}

/** Count the number of trailing zero bits in an expression. The result is
 *  undefined if the value of the expression is zero. */
inline Expr count_trailing_zeros(Expr x) {
    user_assert(x.defined()) << "count trailing zeros of undefined Expr\n";
    Type t = x.type();
    user_assert(t.is_uint() || t.is_int())
        << "Argument to count_trailing_zeros must be an integer\n";
    return Internal::Call::make(t, Internal::Call::count_trailing_zeros,
                                {std::move(x)}, Internal::Call::PureIntrinsic);
}

/** Divide two integers, rounding towards zero. This is the typical
 * behavior of most hardware architectures, which differs from
 * Halide's division operator, which is Euclidean (rounds towards
 * -infinity). */
inline Expr div_round_to_zero(Expr x, Expr y) {
    user_assert(x.defined()) << "div_round_to_zero of undefined dividend\n";
    user_assert(y.defined()) << "div_round_to_zero of undefined divisor\n";
    Internal::match_types(x, y);
    if (x.type().is_uint()) {
        return std::move(x) / std::move(y);
    }
    user_assert(x.type().is_int()) << "First argument to div_round_to_zero is not an integer: " << x << "\n";
    user_assert(y.type().is_int()) << "Second argument to div_round_to_zero is not an integer: " << y << "\n";
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::div_round_to_zero,
                                {std::move(x), std::move(y)},
                                Internal::Call::PureIntrinsic);
}

/** Compute the remainder of dividing two integers, when division is
 * rounding toward zero. This is the typical behavior of most hardware
 * architectures, which differs from Halide's mod operator, which is
 * Euclidean (produces the remainder when division rounds towards
 * -infinity). */
inline Expr mod_round_to_zero(Expr x, Expr y) {
    user_assert(x.defined()) << "mod_round_to_zero of undefined dividend\n";
    user_assert(y.defined()) << "mod_round_to_zero of undefined divisor\n";
    Internal::match_types(x, y);
    if (x.type().is_uint()) {
        return std::move(x) % std::move(y);
    }
    user_assert(x.type().is_int()) << "First argument to mod_round_to_zero is not an integer: " << x << "\n";
    user_assert(y.type().is_int()) << "Second argument to mod_round_to_zero is not an integer: " << y << "\n";
    Type t = x.type();
    return Internal::Call::make(t, Internal::Call::mod_round_to_zero,
                                {std::move(x), std::move(y)},
                                Internal::Call::PureIntrinsic);
}

/** Return a random variable representing a uniformly distributed
 * float in the half-open interval [0.0f, 1.0f). For random numbers of
 * other types, use lerp with a random float as the last parameter.
 *
 * Optionally takes a seed.
 *
 * Note that:
 \code
 Expr x = random_float();
 Expr y = x + x;
 \endcode
 *
 * is very different to
 *
 \code
 Expr y = random_float() + random_float();
 \endcode
 *
 * The first doubles a random variable, and the second adds two
 * independent random variables.
 *
 * A given random variable takes on a unique value that depends
 * deterministically on the pure variables of the function they belong
 * to, the identity of the function itself, and which definition of
 * the function it is used in. They are, however, shared across tuple
 * elements.
 *
 * This function vectorizes cleanly.
 */
inline Expr random_float(Expr seed = Expr()) {
    // Random floats get even IDs
    static std::atomic<int> counter;
    int id = (counter++)*2;

    std::vector<Expr> args;
    if (seed.defined()) {
        user_assert(seed.type() == Int(32))
            << "The seed passed to random_float must have type Int(32), but instead is "
            << seed << " of type " << seed.type() << "\n";
        args.push_back(std::move(seed));
    }
    args.push_back(id);

    // This is (surprisingly) pure - it's a fixed psuedo-random
    // function of its inputs.
    return Internal::Call::make(Float(32), Internal::Call::random,
                                args, Internal::Call::PureIntrinsic);
}

/** Return a random variable representing a uniformly distributed
 * unsigned 32-bit integer. See \ref random_float. Vectorizes cleanly. */
inline Expr random_uint(Expr seed = Expr()) {
    // Random ints get odd IDs
    static std::atomic<int> counter;
    int id = (counter++)*2 + 1;

    std::vector<Expr> args;
    if (seed.defined()) {
        user_assert(seed.type() == Int(32) || seed.type() == UInt(32))
            << "The seed passed to random_int must have type Int(32) or UInt(32), but instead is "
            << seed << " of type " << seed.type() << "\n";
        args.push_back(std::move(seed));
    }
    args.push_back(id);

    return Internal::Call::make(UInt(32), Internal::Call::random,
                                args, Internal::Call::PureIntrinsic);
}

/** Return a random variable representing a uniformly distributed
 * 32-bit integer. See \ref random_float. Vectorizes cleanly. */
inline Expr random_int(Expr seed = Expr()) {
    return cast<int32_t>(random_uint(std::move(seed)));
}

// Secondary args to print can be Exprs or const char *
namespace Internal {
inline NO_INLINE void collect_print_args(std::vector<Expr> &args) {
}

template<typename ...Args>
inline NO_INLINE void collect_print_args(std::vector<Expr> &args, const char *arg, Args&&... more_args) {
    args.push_back(Expr(std::string(arg)));
    collect_print_args(args, std::forward<Args>(more_args)...);
}

template<typename ...Args>
inline NO_INLINE void collect_print_args(std::vector<Expr> &args, Expr arg, Args&&... more_args) {
    args.push_back(std::move(arg));
    collect_print_args(args, std::forward<Args>(more_args)...);
}
}


/** Create an Expr that prints out its value whenever it is
 * evaluated. It also prints out everything else in the arguments
 * list, separated by spaces. This can include string literals. */
//@{
EXPORT Expr print(const std::vector<Expr> &values);

template <typename... Args>
inline NO_INLINE Expr print(Expr a, Args&&... args) {
    std::vector<Expr> collected_args = {std::move(a)};
    Internal::collect_print_args(collected_args, std::forward<Args>(args)...);
    return print(collected_args);
}
//@}

/** Create an Expr that prints whenever it is evaluated, provided that
 * the condition is true. */
// @{
EXPORT Expr print_when(Expr condition, const std::vector<Expr> &values);

template<typename ...Args>
inline NO_INLINE Expr print_when(Expr condition, Expr a, Args&&... args) {
    std::vector<Expr> collected_args = {std::move(a)};
    Internal::collect_print_args(collected_args, std::forward<Args>(args)...);
    return print_when(std::move(condition), collected_args);
}

// @}

/** Create an Expr that that guarantees a precondition.
 * If 'condition' is true, the return value is equal to the first Expr.
 * If 'condition' is false, halide_error() is called, and the return value
 * is arbitrary. Any additional arguments after the first Expr are stringified
 * and passed as a user-facing message to halide_error(), similar to print().
 *
 * Note that this essentially *always* inserts a runtime check into the
 * generated code (except when the condition can be proven at compile time);
 * as such, it should be avoided inside inner loops, except for debugging
 * or testing purposes. Note also that it does not vectorize cleanly (vector
 * values will be scalarized for the check).
 *
 * However, using this to make assertions about (say) input values
 * can be useful, both in terms of correctness and (potentially) in terms
 * of code generation, e.g.
 \code
 Param<int> p;
 Expr y = require(p > 0, p);
 \endcode
 * will allow the optimizer to assume positive, nonzero values for y.
 */
// @{
EXPORT Expr require(Expr condition, const std::vector<Expr> &values);

template<typename ...Args>
inline NO_INLINE Expr require(Expr condition, Expr value, Args&&... args) {
    std::vector<Expr> collected_args = {std::move(value)};
    Internal::collect_print_args(collected_args, std::forward<Args>(args)...);
    return require(std::move(condition), collected_args);
}

// @}


/** Return an undef value of the given type. Halide skips stores that
 * depend on undef values, so you can use this to mean "do not modify
 * this memory location". This is an escape hatch that can be used for
 * several things:
 *
 * You can define a reduction with no pure step, by setting the pure
 * step to undef. Do this only if you're confident that the update
 * steps are sufficient to correctly fill in the domain.
 *
 * For a tuple-valued reduction, you can write an update step that
 * only updates some tuple elements.
 *
 * You can define single-stage pipeline that only has update steps,
 * and depends on the values already in the output buffer.
 *
 * Use this feature with great caution, as you can use it to load from
 * uninitialized memory.
 */
inline Expr undef(Type t) {
    return Internal::Call::make(t, Internal::Call::undef,
                                std::vector<Expr>(),
                                Internal::Call::PureIntrinsic);
}

template<typename T>
inline Expr undef() {
    return undef(type_of<T>());
}

namespace Internal {
EXPORT Expr memoize_tag_helper(Expr result, const std::vector<Expr> &cache_key_values);
}  // namespace Internal

/** Control the values used in the memoization cache key for memoize.
 * Normally parameters and other external dependencies are
 * automatically inferred and added to the cache key. The memoize_tag
 * operator allows computing one expression and using either the
 * computed value, or one or more other expressions in the cache key
 * instead of the parameter dependencies of the computation. The
 * single argument version is completely safe in that the cache key
 * will use the actual computed value -- it is difficult or imposible
 * to produce erroneous caching this way. The more-than-one argument
 * version allows generating cache keys that do not uniquely identify
 * the computation and thus can result in caching errors.
 *
 * A potential use for the single argument version is to handle a
 * floating-point parameter that is quantized to a small
 * integer. Mutliple values of the float will produce the same integer
 * and moving the caching to using the integer for the key is more
 * efficient.
 *
 * The main use for the more-than-one argument version is to provide
 * cache key information for Handles and ImageParams, which otherwise
 * are not allowed inside compute_cached operations. E.g. when passing
 * a group of parameters to an external array function via a Handle,
 * memoize_tag can be used to isolate the actual values used by that
 * computation. If an ImageParam is a constant image with a persistent
 * digest, memoize_tag can be used to key computations using that image
 * on the digest. */
// @{
template<typename ...Args>
inline NO_INLINE Expr memoize_tag(Expr result, Args&&... args) {
    std::vector<Expr> collected_args{std::forward<Args>(args)...};
    return Internal::memoize_tag_helper(std::move(result), collected_args);
}
// @}

/** Expressions tagged with this intrinsic are considered to be part
 * of the steady state of some loop with a nasty beginning and end
 * (e.g. a boundary condition). When Halide encounters likely
 * intrinsics, it splits the containing loop body into three, and
 * tries to simplify down all conditions that lead to the likely. For
 * example, given the expression: select(x < 1, bar, x > 10, bar,
 * likely(foo)), Halide will split the loop over x into portions where
 * x < 1, 1 <= x <= 10, and x > 10.
 *
 * You're unlikely to want to call this directly. You probably want to
 * use the boundary condition helpers in the BoundaryConditions
 * namespace instead.
 */
inline Expr likely(Expr e) {
    Type t = e.type();
    return Internal::Call::make(t, Internal::Call::likely,
                                {std::move(e)}, Internal::Call::PureIntrinsic);
}

/** Equivalent to likely, but only triggers a loop partitioning if
 * found in an innermost loop. */
inline Expr likely_if_innermost(Expr e) {
    Type t = e.type();
    return Internal::Call::make(t, Internal::Call::likely_if_innermost,
                                {std::move(e)}, Internal::Call::PureIntrinsic);
}


/** Cast an expression to the halide type corresponding to the C++
 * type T clamping to the minimum and maximum values of the result
 * type. */
template <typename T>
Expr saturating_cast(Expr e) {
    return saturating_cast(type_of<T>(), std::move(e));
}

/** Cast an expression to a new type, clamping to the minimum and
 * maximum values of the result type. */
EXPORT Expr saturating_cast(Type t, Expr e);

}

#endif
#ifndef HALIDE_SCOPE_H
#define HALIDE_SCOPE_H

#include <string>
#include <map>
#include <stack>
#include <utility>
#include <iostream>


/** \file
 * Defines the Scope class, which is used for keeping track of names in a scope while traversing IR
 */

namespace Halide {
namespace Internal {

/** A stack which can store one item very efficiently. Using this
 * instead of std::stack speeds up Scope substantially. */
template<typename T>
class SmallStack {
private:
    T _top;
    std::vector<T> _rest;
    bool _empty;

public:
    SmallStack() : _empty(true) {}

    void pop() {
        if (_rest.empty()) {
            _empty = true;
            _top = T();
        } else {
            _top = _rest.back();
            _rest.pop_back();
        }
    }

    void push(const T &t) {
        if (_empty) {
            _empty = false;
        } else {
            _rest.push_back(_top);
        }
        _top = t;
    }

    T top() const {
        return _top;
    }

    T &top_ref() {
        return _top;
    }

    const T &top_ref() const {
        return _top;
    }

    bool empty() const {
        return _empty;
    }
};

/** A common pattern when traversing Halide IR is that you need to
 * keep track of stuff when you find a Let or a LetStmt, and that it
 * should hide previous values with the same name until you leave the
 * Let or LetStmt nodes This class helps with that. */
template<typename T>
class Scope {
private:
    std::map<std::string, SmallStack<T>> table;

    // Copying a scope object copies a large table full of strings and
    // stacks. Bad idea.
    Scope(const Scope<T> &);
    Scope<T> &operator=(const Scope<T> &);

    const Scope<T> *containing_scope;


public:
    Scope() : containing_scope(nullptr) {}

    /** Set the parent scope. If lookups fail in this scope, they
     * check the containing scope before returning an error. Caller is
     * responsible for managing the memory of the containing scope. */
    void set_containing_scope(const Scope<T> *s) {
        containing_scope = s;
    }

    /** A const ref to an empty scope. Useful for default function
     * arguments, which would otherwise require a copy constructor
     * (with llvm in c++98 mode) */
    static const Scope<T> &empty_scope() {
        static Scope<T> *_empty_scope = new Scope<T>();
        return *_empty_scope;
    }

    /** Retrieve the value referred to by a name */
    T get(const std::string &name) const {
        typename std::map<std::string, SmallStack<T>>::const_iterator iter = table.find(name);
        if (iter == table.end() || iter->second.empty()) {
            if (containing_scope) {
                return containing_scope->get(name);
            } else {
                internal_error << "Symbol '" << name << "' not found\n";
            }
        }
        return iter->second.top();
    }

    /** Return a reference to an entry. Does not consider the containing scope. */
    T &ref(const std::string &name) {
        typename std::map<std::string, SmallStack<T>>::iterator iter = table.find(name);
        if (iter == table.end() || iter->second.empty()) {
            internal_error << "Symbol '" << name << "' not found\n";
        }
        return iter->second.top_ref();
    }

    /** Tests if a name is in scope */
    bool contains(const std::string &name) const {
        typename std::map<std::string, SmallStack<T>>::const_iterator iter = table.find(name);
        if (iter == table.end() || iter->second.empty()) {
            if (containing_scope) {
                return containing_scope->contains(name);
            } else {
                return false;
            }
        }
        return true;
    }

    /** Add a new (name, value) pair to the current scope. Hide old
     * values that have this name until we pop this name.
     */
    void push(const std::string &name, const T &value) {
        table[name].push(value);
    }

    /** A name goes out of scope. Restore whatever its old value
     * was (or remove it entirely if there was nothing else of the
     * same name in an outer scope) */
    void pop(const std::string &name) {
        typename std::map<std::string, SmallStack<T>>::iterator iter = table.find(name);
        internal_assert(iter != table.end()) << "Name not in symbol table: " << name << "\n";
        iter->second.pop();
        if (iter->second.empty()) {
            table.erase(iter);
        }
    }

    /** Iterate through the scope. Does not capture any containing scope. */
    class const_iterator {
        typename std::map<std::string, SmallStack<T>>::const_iterator iter;
    public:
        explicit const_iterator(const typename std::map<std::string, SmallStack<T>>::const_iterator &i) :
            iter(i) {
        }

        const_iterator() {}

        bool operator!=(const const_iterator &other) {
            return iter != other.iter;
        }

        void operator++() {
            ++iter;
        }

        const std::string &name() {
            return iter->first;
        }

        const SmallStack<T> &stack() {
            return iter->second;
        }

        const T &value() {
            return iter->second.top_ref();
        }
    };

    const_iterator cbegin() const {
        return const_iterator(table.begin());
    }

    const_iterator cend() const {
        return const_iterator(table.end());
    }

    class iterator {
        typename std::map<std::string, SmallStack<T>>::iterator iter;
    public:
        explicit iterator(typename std::map<std::string, SmallStack<T>>::iterator i) :
            iter(i) {
        }

        iterator() {}

        bool operator!=(const iterator &other) {
            return iter != other.iter;
        }

        void operator++() {
            ++iter;
        }

        const std::string &name() {
            return iter->first;
        }

        SmallStack<T> &stack() {
            return iter->second;
        }

        T &value() {
            return iter->second.top_ref();
        }
    };

    iterator begin() {
        return iterator(table.begin());
    }

    iterator end() {
        return iterator(table.end());
    }

    void swap(Scope<T> &other) {
        table.swap(other.table);
        std::swap(containing_scope, other.containing_scope);
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &stream, const Scope<T>& s) {
    stream << "{\n";
    typename Scope<T>::const_iterator iter;
    for (iter = s.cbegin(); iter != s.cend(); ++iter) {
        stream << "  " << iter.name() << "\n";
    }
    stream << "}";
    return stream;
}

}
}

#endif
#ifndef HALIDE_INTERVAL_H
#define HALIDE_INTERVAL_H

/** \file
 * Defines the Interval class
 */


namespace Halide {
namespace Internal {

/** A class to represent ranges of Exprs. Can be unbounded above or below. */
struct Interval {

    /** Exprs to represent positive and negative infinity */
    static Expr pos_inf, neg_inf;

    /** The lower and upper bound of the interval. They are included
     * in the interval. */
    Expr min, max;

    /** A default-constructed Interval is everything */
    Interval() : min(neg_inf), max(pos_inf) {}

    /** Construct an interval from a lower and upper bound. */
    Interval(Expr min, Expr max) : min(min), max(max) {
        internal_assert(min.defined() && max.defined());
    }

    /** The interval representing everything. */
    static Interval everything() {return Interval(neg_inf, pos_inf);}

    /** The interval representing nothing. */
    static Interval nothing() {return Interval(pos_inf, neg_inf);}

    /** Construct an interval representing a single point */
    static Interval single_point(Expr e) {return Interval(e, e);}

    /** Is the interval the empty set */
    bool is_empty() const {return min.same_as(pos_inf) || max.same_as(neg_inf);}

    /** Is the interval the entire range */
    bool is_everything() const {return min.same_as(neg_inf) && max.same_as(pos_inf);}

    /** Is the interval just a single value (min == max) */
    bool is_single_point() const {return min.same_as(max);}

    /** Is the interval a particular single value */
    bool is_single_point(Expr e) const {return min.same_as(e) && max.same_as(e);}

    /** Does the interval have a finite least upper bound */
    bool has_upper_bound() const {return !max.same_as(pos_inf) && !is_empty();}

    /** Does the interval have a finite greatest lower bound */
    bool has_lower_bound() const {return !min.same_as(neg_inf) && !is_empty();}

    /** Does the interval have a finite upper and lower bound */
    bool is_bounded() const {return has_upper_bound() && has_lower_bound();}

    /** Is the interval the same as another interval */
    bool same_as(const Interval &other) {return min.same_as(other.min) && max.same_as(other.max);}

    /** Expand the interval to include another Interval */
    EXPORT void include(const Interval &i);

    /** Expand the interval to include an Expr */
    EXPORT void include(Expr e);

    /** Construct the smallest interval containing two intervals. */
    EXPORT static Interval make_union(const Interval &a, const Interval &b);

    /** Construct the largest interval contained within two intervals. */
    EXPORT static Interval make_intersection(const Interval &a, const Interval &b);

    /** An eagerly-simplifying max of two Exprs that respects infinities. */
    EXPORT static Expr make_max(Expr a, Expr b);

    /** An eagerly-simplifying min of two Exprs that respects infinities. */
    EXPORT static Expr make_min(Expr a, Expr b);

    bool operator==(const Interval &other) const {
        return (min.same_as(other.min)) && (max.same_as(other.max));
    }
};

EXPORT void interval_test();

}
}

#endif

namespace Halide {
namespace Internal {

typedef std::map<std::pair<std::string, int>, Interval> FuncValueBounds;

/** Given an expression in some variables, and a map from those
 * variables to their bounds (in the form of (minimum possible value,
 * maximum possible value)), compute two expressions that give the
 * minimum possible value and the maximum possible value of this
 * expression. Max or min may be undefined expressions if the value is
 * not bounded above or below. If the expression is a vector, also
 * takes the bounds across the vector lanes and returns a scalar
 * result.
 *
 * This is for tasks such as deducing the region of a buffer
 * loaded by a chunk of code.
 */
Interval bounds_of_expr_in_scope(Expr expr,
                                 const Scope<Interval> &scope,
                                 const FuncValueBounds &func_bounds = FuncValueBounds(),
                                 bool const_bound = false);

/* Given a varying expression, try to find a constant that is either:
 * An upper bound (always greater than or equal to the expression), or
 * A lower bound (always less than or equal to the expression)
 * If it fails, returns an undefined Expr. */
enum class Direction {Upper, Lower};
Expr find_constant_bound(Expr e, Direction d,
                         const Scope<Interval> &scope = Scope<Interval>());

/** Represents the bounds of a region of arbitrary dimension. Zero
 * dimensions corresponds to a scalar region. */
struct Box {
    /** The conditions under which this region may be touched. */
    Expr used;

    /** The bounds if it is touched. */
    std::vector<Interval> bounds;

    Box() {}
    Box(size_t sz) : bounds(sz) {}
    Box(const std::vector<Interval> &b) : bounds(b) {}

    size_t size() const {return bounds.size();}
    bool empty() const {return bounds.empty();}
    Interval &operator[](int i) {return bounds[i];}
    const Interval &operator[](int i) const {return bounds[i];}
    void resize(size_t sz) {bounds.resize(sz);}
    void push_back(const Interval &i) {bounds.push_back(i);}

    /** Check if the used condition is defined and not trivially true. */
    bool maybe_unused() const {return used.defined() && !is_one(used);}

    friend std::ostream& operator<<(std::ostream& stream, const Box& b) {
        stream << "{";
        for (size_t dim = 0; dim < b.size(); dim++) {
            if (dim > 0) {
                stream << ", ";
            }
            stream << "[" << b[dim].min << ", " << b[dim].max << "]";
        }
        stream << "}";
        return stream;
    }
};

/** Expand box a to encompass box b */
void merge_boxes(Box &a, const Box &b);

/** Test if box a could possibly overlap box b. */
bool boxes_overlap(const Box &a, const Box &b);

/** The union of two boxes */
Box box_union(const Box &a, const Box &b);

/** The intersection of two boxes */
Box box_intersection(const Box &a, const Box &b);

/** Test if box a provably contains box b */
bool box_contains(const Box &a, const Box &b);


/** Compute rectangular domains large enough to cover all the 'Call's
 * to each function that occurs within a given statement or
 * expression. This is useful for figuring out what regions of things
 * to evaluate. */
// @{
std::map<std::string, Box> boxes_required(Expr e,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
std::map<std::string, Box> boxes_required(Stmt s,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Compute rectangular domains large enough to cover all the
 * 'Provides's to each function that occurs within a given statement
 * or expression. */
// @{
std::map<std::string, Box> boxes_provided(Expr e,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
std::map<std::string, Box> boxes_provided(Stmt s,
                                          const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                          const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Compute rectangular domains large enough to cover all the 'Call's
 * and 'Provides's to each function that occurs within a given
 * statement or expression. */
// @{
std::map<std::string, Box> boxes_touched(Expr e,
                                         const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                         const FuncValueBounds &func_bounds = FuncValueBounds());
std::map<std::string, Box> boxes_touched(Stmt s,
                                         const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                                         const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Variants of the above that are only concerned with a single function. */
// @{
Box box_required(Expr e, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());
Box box_required(Stmt s, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());

Box box_provided(Expr e, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());
Box box_provided(Stmt s, std::string fn,
                 const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                 const FuncValueBounds &func_bounds = FuncValueBounds());

Box box_touched(Expr e, std::string fn,
                const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                const FuncValueBounds &func_bounds = FuncValueBounds());
Box box_touched(Stmt s, std::string fn,
                const Scope<Interval> &scope = Scope<Interval>::empty_scope(),
                const FuncValueBounds &func_bounds = FuncValueBounds());
// @}

/** Compute the maximum and minimum possible value for each function
 * in an environment. */
FuncValueBounds compute_function_value_bounds(const std::vector<std::string> &order,
                                              const std::map<std::string, Function> &env);

EXPORT void bounds_test();

}
}

#endif

#include <map>

namespace Halide {

struct Target;

namespace Internal {

/** Insert checks to make sure a statement doesn't read out of bounds
 * on inputs or outputs, and that the inputs and outputs conform to
 * the format required (e.g. stride.0 must be 1).
 */
Stmt add_image_checks(Stmt s,
                      const std::vector<Function> &outputs,
                      const Target &t,
                      const std::vector<std::string> &order,
                      const std::map<std::string, Function> &env,
                      const FuncValueBounds &fb);


}
}

#endif
#ifndef HALIDE_INTERNAL_ADD_PARAMETER_CHECKS_H
#define HALIDE_INTERNAL_ADD_PARAMETER_CHECKS_H

/** \file
 *
 * Defines the lowering pass that adds the assertions that validate
 * scalar parameters.
 */


namespace Halide {

struct Target;

namespace Internal {

/** Insert checks to make sure that all referenced parameters meet
 * their constraints. */
Stmt add_parameter_checks(Stmt s, const Target &t);


}
}

#endif
#ifndef HALIDE_ALIGN_LOADS_H
#define HALIDE_ALIGN_LOADS_H

/** \file
 * Defines a lowering pass that rewrites unaligned loads into
 * sequences of aligned loads.
 */
namespace Halide {
namespace Internal {

/** Attempt to rewrite unaligned loads from buffers which are known to
 * be aligned to instead load aligned vectors that cover the original
 * load, and then slice the original load out of the aligned
 * vectors. */
Stmt align_loads(Stmt s, int alignment);

}
}

#endif
#ifndef HALIDE_ALLOCATION_BOUNDS_INFERENCE_H
#define HALIDE_ALLOCATION_BOUNDS_INFERENCE_H

/** \file
 * Defines the lowering pass that determines how large internal allocations should be.
 */


namespace Halide {
namespace Internal {

/** Take a partially statement with Realize nodes in terms of
 * variables, and define values for those variables. */
Stmt allocation_bounds_inference(Stmt s,
                                 const std::map<std::string, Function> &env,
                                 const std::map<std::pair<std::string, int>, Interval> &func_bounds);
}
}

#endif
#ifndef APPLY_SPLIT_H
#define APPLY_SPLIT_H

/** \file
 *
 * Defines method that returns a list of let stmts, substitutions, and
 * predicates to be added given a split schedule.
 */

#include <map>
#include <utility>
#include <vector>


namespace Halide {
namespace Internal {

struct ApplySplitResult {
    // If type is "Substitution", then this represents a substitution of
    // variable "name" to value. If type is "LetStmt", we should insert a new
    // let stmt defining "name" with value "value". If type is "Predicate", we
    // should ignore "name" and the predicate is "value".

    std::string name;
    Expr value;

    enum Type {Substitution = 0, LetStmt, Predicate};
    Type type;

    ApplySplitResult(const std::string &n, Expr val, Type t)
        : name(n), value(val), type(t) {}
    ApplySplitResult(Expr val) : name(""), value(val), type(Predicate) {}

    bool is_substitution() const {return (type == Substitution);}
    bool is_let() const {return (type == LetStmt);}
    bool is_predicate() const {return (type == Predicate);}
};

/** Given a Split schedule on a definition (init or update), return a list of
 * of predicates on the definition, substitutions that needs to be applied to
 * the definition (in ascending order of application), and let stmts which
 * defined the values of variables referred by the predicates and substitutions
 * (ordered from innermost to outermost let). */
std::vector<ApplySplitResult> apply_split(
    const Split &split, bool is_update, std::string prefix,
    std::map<std::string, Expr> &dim_extent_alignment);

/** Compute the loop bounds of the new dimensions resulting from applying the
 * split schedules using the loop bounds of the old dimensions. */
std::vector<std::pair<std::string, Expr>> compute_loop_bounds_after_split(
    const Split &split, std::string prefix);

}
}

#endif
#ifndef HALIDE_ARGUMENT_H
#define HALIDE_ARGUMENT_H

/** \file
 * Defines a type used for expressing the type signature of a
 * generated halide pipeline
 */


namespace Halide {

/**
 * A struct representing an argument to a halide-generated
 * function. Used for specifying the function signature of
 * generated code.
 */
struct Argument {
    /** The name of the argument */
    std::string name;

    /** An argument is either a primitive type (for parameters), or a
     * buffer pointer.
     *
     * If kind == InputScalar, then type fully encodes the expected type
     * of the scalar argument.
     *
     * If kind == InputBuffer|OutputBuffer, then type.bytes() should be used
     * to determine* elem_size of the buffer; additionally, type.code *should*
     * reflect the expected interpretation of the buffer data (e.g. float vs int),
     * but there is no runtime enforcement of this at present.
     */
    enum Kind {
        InputScalar = halide_argument_kind_input_scalar,
        InputBuffer = halide_argument_kind_input_buffer,
        OutputBuffer = halide_argument_kind_output_buffer
    };
    Kind kind;

    /** If kind == InputBuffer|OutputBuffer, this is the dimensionality of the buffer.
     * If kind == InputScalar, this value is ignored (and should always be set to zero) */
    uint8_t dimensions;

    /** If this is a scalar parameter, then this is its type.
     *
     * If this is a buffer parameter, this this is the type of its
     * elements.
     *
     * Note that type.lanes should always be 1 here. */
    Type type;

    /** If this is a scalar parameter, then these are its default, min, max values.
     * By default, they are left unset, implying "no default, no min, no max". */
    Expr def, min, max;

    Argument() : kind(InputScalar), dimensions(0) {}
    Argument(const std::string &_name, Kind _kind, const Type &_type, int _dimensions,
             Expr _def = Expr(),
             Expr _min = Expr(),
             Expr _max = Expr()) :
        name(_name), kind(_kind), dimensions((uint8_t) _dimensions), type(_type), def(_def), min(_min), max(_max) {
        internal_assert(_dimensions >= 0 && _dimensions <= 255);
        user_assert(!(is_scalar() && dimensions != 0))
            << "Scalar Arguments must specify dimensions of 0";
        user_assert(!(is_buffer() && def.defined()))
            << "Scalar default must not be defined for Buffer Arguments";
        user_assert(!(is_buffer() && min.defined()))
            << "Scalar min must not be defined for Buffer Arguments";
        user_assert(!(is_buffer() && max.defined()))
            << "Scalar max must not be defined for Buffer Arguments";
    }

    template<typename T>
    Argument(Buffer<T> im) :
        name(im.name()),
        kind(InputBuffer),
        dimensions(im.dimensions()),
        type(im.type()) {}

    bool is_buffer() const { return kind == InputBuffer || kind == OutputBuffer; }
    bool is_scalar() const { return kind == InputScalar; }

    bool is_input() const { return kind == InputScalar || kind == InputBuffer; }
    bool is_output() const { return kind == OutputBuffer; }

    bool operator==(const Argument &rhs) const {
        return name == rhs.name &&
               kind == rhs.kind &&
               dimensions == rhs.dimensions &&
               type == rhs.type &&
               def.same_as(rhs.def) &&
               min.same_as(rhs.min) &&
               max.same_as(rhs.max);
    }
};

}

#endif
#ifndef HALIDE_ASSOCIATIVE_OPS_TABLE_H
#define HALIDE_ASSOCIATIVE_OPS_TABLE_H

/** \file
 * Tables listing associative operators and their identities.
 */

#ifndef HALIDE_IR_EQUALITY_H
#define HALIDE_IR_EQUALITY_H

/** \file
 * Methods to test Exprs and Stmts for equality of value
 */


namespace Halide {
namespace Internal {

/** A compare struct suitable for use in std::map and std::set that
 * computes a lexical ordering on IR nodes. */
struct IRDeepCompare {
    EXPORT bool operator()(const Expr &a, const Expr &b) const;
    EXPORT bool operator()(const Stmt &a, const Stmt &b) const;
};

/** Lossily track known equal exprs with a cache. On collision, the
 * old pair is evicted. Used below by ExprWithCompareCache. */
class IRCompareCache {
private:
    struct Entry {
        Expr a, b;
    };

    int bits;

    uint32_t hash(const Expr &a, const Expr &b) const {
        // Note this hash is symmetric in a and b, so that a
        // comparison in a and b hashes to the same bucket as
        // a comparison on b and a.
        uint64_t pa = (uint64_t)(a.get());
        uint64_t pb = (uint64_t)(b.get());
        uint64_t mix = (pa + pb) + (pa ^ pb);
        mix ^= (mix >> bits);
        mix ^= (mix >> (bits*2));
        uint32_t bottom = mix & ((1 << bits) - 1);
        return bottom;
    }

    std::vector<Entry> entries;

public:
    void insert(const Expr &a, const Expr &b) {
        uint32_t h = hash(a, b);
        entries[h].a = a;
        entries[h].b = b;
    }

    bool contains(const Expr &a, const Expr &b) const {
        uint32_t h = hash(a, b);
        const Entry &e = entries[h];
        return ((a.same_as(e.a) && b.same_as(e.b)) ||
                (a.same_as(e.b) && b.same_as(e.a)));
    }

    void clear() {
        for (size_t i = 0; i < entries.size(); i++) {
            entries[i].a = Expr();
            entries[i].b = Expr();
        }
    }

    IRCompareCache() {}
    IRCompareCache(int b) : bits(b), entries(static_cast<size_t>(1) << bits) {}
};

/** A wrapper about Exprs so that they can be deeply compared with a
 * cache for known-equal subexpressions. Useful for unsanitized Exprs
 * coming in from the front-end, which may be horrible graphs with
 * sub-expressions that are equal by value but not by identity. This
 * isn't a comparison object like IRDeepCompare above, because libc++
 * requires that comparison objects be stateless (and constructs a new
 * one for each comparison!), so they can't have a cache associated
 * with them. However, by sneakily making the cache a mutable member
 * of the objects being compared, we can dodge this issue.
 *
 * Clunky example usage:
 *
\code
Expr a, b, c, query;
std::set<ExprWithCompareCache> s;
IRCompareCache cache(8);
s.insert(ExprWithCompareCache(a, &cache));
s.insert(ExprWithCompareCache(b, &cache));
s.insert(ExprWithCompareCache(c, &cache));
if (m.contains(ExprWithCompareCache(query, &cache))) {...}
\endcode
 *
 */
struct ExprWithCompareCache {
    Expr expr;
    mutable IRCompareCache *cache;

    ExprWithCompareCache() : cache(nullptr) {}
    ExprWithCompareCache(const Expr &e, IRCompareCache *c) : expr(e), cache(c) {}

    /** The comparison uses (and updates) the cache */
    EXPORT bool operator<(const ExprWithCompareCache &other) const;
};

/** Compare IR nodes for equality of value. Traverses entire IR
 * tree. For equality of reference, use Expr::same_as. If you're
 * comparing non-CSE'd Exprs, use graph_equal, which is safe for nasty
 * graphs of IR nodes. */
// @{
EXPORT bool equal(const Expr &a, const Expr &b);
EXPORT bool equal(const Stmt &a, const Stmt &b);
EXPORT bool graph_equal(const Expr &a, const Expr &b);
EXPORT bool graph_equal(const Stmt &a, const Stmt &b);
// @}



EXPORT void ir_equality_test();

}
}

#endif

#include <iostream>
#include <vector>

namespace Halide {
namespace Internal {

/**
 * Represent an associative op with its identity. The op may be multi-dimensional,
 * e.g. complex multiplication. 'is_commutative' is set to true if the op is also
 * commutative in addition to being associative.
 *
 * For example, complex multiplication is represented as:
 \code
 AssociativePattern pattern(
    {x0 * y0 - x1 * y1, x1 * y0 + x0 * y1},
    {one, zero},
    true
 );
 \endcode
 */
struct AssociativePattern {
    /** Contain the binary operators for each dimension of the associative op. */
    std::vector<Expr> ops;
    /** Contain the identities for each dimension of the associative op. */
    std::vector<Expr> identities;
    /** Indicate if the associative op is also commutative. */
    bool is_commutative;

    AssociativePattern() : is_commutative(false) {}
    AssociativePattern(size_t size) : ops(size), identities(size), is_commutative(false) {}
    AssociativePattern(const std::vector<Expr> &ops, const std::vector<Expr> &ids, bool is_commutative)
        : ops(ops), identities(ids), is_commutative(is_commutative) {}
    AssociativePattern(Expr op, Expr id, bool is_commutative)
        : ops({op}), identities({id}), is_commutative(is_commutative) {}

    bool operator==(const AssociativePattern &other) const {
        if ((is_commutative != other.is_commutative) || (ops.size() != other.ops.size())) {
            return false;
        }
        for (size_t i = 0; i < size(); ++i) {
            if (!equal(ops[i], other.ops[i]) || !equal(identities[i], other.identities[i])) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const AssociativePattern &other) const { return !(*this == other); }
    size_t size() const { return ops.size(); }
    bool commutative() const { return is_commutative; }
};

const std::vector<AssociativePattern> &get_ops_table(const std::vector<Expr> &exprs);

}
}

#endif
#ifndef HALIDE_ASSOCIATIVITY_H
#define HALIDE_ASSOCIATIVITY_H

/** \file
 *
 * Methods for extracting an associative operator from a Func's update definition
 * if there is any and computing the identity of the associative operator.
 */


#include <functional>

namespace Halide {
namespace Internal {

/**
 * Represent the equivalent associative op of an update definition.
 * For example, the following associative Expr, min(f(x), g(r.x) + 2),
 * where f(x) is the self-recurrence term, is represented as:
 \code
 AssociativeOp assoc(
    AssociativePattern(min(x, y), +inf, true),
    {Replacement("x", f(x))},
    {Replacement("y", g(r.x) + 2)},
    true
 );
 \endcode
 *
 * 'pattern' contains the list of equivalent binary/unary operators (+ identities)
 * for each Tuple element in the update definition. 'pattern' also contains
 * a boolean that indicates if the op is also commutative. 'xs' and 'ys'
 * contain the corresponding definition of each variable in the list of
 * binary operators.
 *
 * For unary operator, 'xs' is not set, i.e. it will be a pair of empty string
 * and undefined Expr: {"", Expr()}. 'pattern' will only contain the 'y' term in
 * this case. For example, min(g(r.x), 4), will be represented as:
 \code
 AssociativeOp assoc(
    AssociativePattern(y, 0, false),
    {Replacement("", Expr())},
    {Replacement("y", min(g(r.x), 4))},
    true
 );
 \endcode
 *
 * Self-assignment, f(x) = f(x), will be represented as:
 \code
 AssociativeOp assoc(
    AssociativePattern(x, 0, true),
    {Replacement("x", f(x))},
    {Replacement("", Expr())},
    true
 );
 \endcode
 * For both unary operator and self-assignment cases, the identity does not
 * matter. It can be anything.
 */
struct AssociativeOp {
    struct Replacement {
        /** Variable name that is used to replace the expr in 'op'. */
        std::string var;
        Expr expr;

        Replacement() {}
        Replacement(const std::string &var, Expr expr) : var(var), expr(expr) {}

        bool operator==(const Replacement &other) const {
            return (var == other.var) && equal(expr, other.expr);
        }
        bool operator!=(const Replacement &other) const {
            return !(*this == other);
        }
    };

    /** List of pairs of binary associative op and its identity. */
    AssociativePattern pattern;
    std::vector<Replacement> xs;
    std::vector<Replacement> ys;
    bool is_associative;

    AssociativeOp() : is_associative(false) {}
    AssociativeOp(size_t size) : pattern(size), xs(size), ys(size), is_associative(false) {}
    AssociativeOp(const AssociativePattern &p, const std::vector<Replacement> &xs,
                  const std::vector<Replacement> &ys, bool is_associative)
        : pattern(p), xs(xs), ys(ys), is_associative(is_associative) {}

    bool associative() const { return is_associative; }
    bool commutative() const { return pattern.is_commutative; }
    size_t size() const { return pattern.size(); }
};

/**
 * Given an update definition of a Func 'f', determine its equivalent
 * associative binary/unary operator if there is any. 'is_associative'
 * indicates if the operation was successfuly proven as associative.
 */
AssociativeOp prove_associativity(
    const std::string &f, std::vector<Expr> args, std::vector<Expr> exprs);

EXPORT void associativity_test();

}
}

#endif
#ifndef HALIDE_INTERNAL_AUTO_SCHEDULE_H
#define HALIDE_INTERNAL_AUTO_SCHEDULE_H

/** \file
 *
 * Defines the method that does automatic scheduling of Funcs within a pipeline.
 */


namespace Halide {

/** A struct representing the machine parameters to generate the auto-scheduled
 * code for. */
struct MachineParams {
    /** Maximum level of parallelism avalaible. */
    Expr parallelism;
    /** Size of the last-level cache (in KB). */
    Expr last_level_cache_size;
    /** Indicates how much more expensive is the cost of a load compared to
     * the cost of an arithmetic operation at last level cache. */
    Expr balance;

    explicit MachineParams(int32_t parallelism, int32_t llc, int32_t balance)
        : parallelism(parallelism), last_level_cache_size(llc), balance(balance) {}
};

namespace Internal {

/** Generate schedules for Funcs within a pipeline. The Funcs should not already
 * have specializations or schedules as the current auto-scheduler does not take
 * into account user-defined schedules or specializations. This applies the
 * schedules and returns a string representation of the schedules. The target
 * architecture is specified by 'target'. */
EXPORT std::string generate_schedules(const std::vector<Function> &outputs,
                                      const Target &target,
                                      const MachineParams &arch_params);

}
}

#endif
#ifndef HALIDE_INTERNAL_AUTO_SCHEDULE_UTILS_H
#define HALIDE_INTERNAL_AUTO_SCHEDULE_UTILS_H

/** \file
 *
 * Defines util functions that used by auto scheduler.
 */

#include <set>
#include <limits>

#ifndef HALIDE_IR_VISITOR_H
#define HALIDE_IR_VISITOR_H


#include <set>
#include <map>
#include <string>

/** \file
 * Defines the base class for things that recursively walk over the IR
 */

namespace Halide {
namespace Internal {

/** A base class for algorithms that need to recursively walk over the
 * IR. The default implementations just recursively walk over the
 * children. Override the ones you care about.
 */
class IRVisitor {
public:
    EXPORT virtual ~IRVisitor();
    EXPORT virtual void visit(const IntImm *);
    EXPORT virtual void visit(const UIntImm *);
    EXPORT virtual void visit(const FloatImm *);
    EXPORT virtual void visit(const StringImm *);
    EXPORT virtual void visit(const Cast *);
    EXPORT virtual void visit(const Variable *);
    EXPORT virtual void visit(const Add *);
    EXPORT virtual void visit(const Sub *);
    EXPORT virtual void visit(const Mul *);
    EXPORT virtual void visit(const Div *);
    EXPORT virtual void visit(const Mod *);
    EXPORT virtual void visit(const Min *);
    EXPORT virtual void visit(const Max *);
    EXPORT virtual void visit(const EQ *);
    EXPORT virtual void visit(const NE *);
    EXPORT virtual void visit(const LT *);
    EXPORT virtual void visit(const LE *);
    EXPORT virtual void visit(const GT *);
    EXPORT virtual void visit(const GE *);
    EXPORT virtual void visit(const And *);
    EXPORT virtual void visit(const Or *);
    EXPORT virtual void visit(const Not *);
    EXPORT virtual void visit(const Select *);
    EXPORT virtual void visit(const Load *);
    EXPORT virtual void visit(const Ramp *);
    EXPORT virtual void visit(const Broadcast *);
    EXPORT virtual void visit(const Call *);
    EXPORT virtual void visit(const Let *);
    EXPORT virtual void visit(const LetStmt *);
    EXPORT virtual void visit(const AssertStmt *);
    EXPORT virtual void visit(const ProducerConsumer *);
    EXPORT virtual void visit(const For *);
    EXPORT virtual void visit(const Store *);
    EXPORT virtual void visit(const Provide *);
    EXPORT virtual void visit(const Allocate *);
    EXPORT virtual void visit(const Free *);
    EXPORT virtual void visit(const Realize *);
    EXPORT virtual void visit(const Block *);
    EXPORT virtual void visit(const IfThenElse *);
    EXPORT virtual void visit(const Evaluate *);
    EXPORT virtual void visit(const Shuffle *);
    EXPORT virtual void visit(const Prefetch *);
};

/** A base class for algorithms that walk recursively over the IR
 * without visiting the same node twice. This is for passes that are
 * capable of interpreting the IR as a DAG instead of a tree. */
class IRGraphVisitor : public IRVisitor {
protected:
    /** By default these methods add the node to the visited set, and
     * return whether or not it was already there. If it wasn't there,
     * it delegates to the appropriate visit method. You can override
     * them if you like. */
    // @{
    EXPORT virtual void include(const Expr &);
    EXPORT virtual void include(const Stmt &);
    // @}

    /** The nodes visited so far */
    std::set<const IRNode *> visited;

public:

    /** These methods should call 'include' on the children to only
     * visit them if they haven't been visited already. */
    // @{
    EXPORT virtual void visit(const IntImm *);
    EXPORT virtual void visit(const UIntImm *);
    EXPORT virtual void visit(const FloatImm *);
    EXPORT virtual void visit(const StringImm *);
    EXPORT virtual void visit(const Cast *);
    EXPORT virtual void visit(const Variable *);
    EXPORT virtual void visit(const Add *);
    EXPORT virtual void visit(const Sub *);
    EXPORT virtual void visit(const Mul *);
    EXPORT virtual void visit(const Div *);
    EXPORT virtual void visit(const Mod *);
    EXPORT virtual void visit(const Min *);
    EXPORT virtual void visit(const Max *);
    EXPORT virtual void visit(const EQ *);
    EXPORT virtual void visit(const NE *);
    EXPORT virtual void visit(const LT *);
    EXPORT virtual void visit(const LE *);
    EXPORT virtual void visit(const GT *);
    EXPORT virtual void visit(const GE *);
    EXPORT virtual void visit(const And *);
    EXPORT virtual void visit(const Or *);
    EXPORT virtual void visit(const Not *);
    EXPORT virtual void visit(const Select *);
    EXPORT virtual void visit(const Load *);
    EXPORT virtual void visit(const Ramp *);
    EXPORT virtual void visit(const Broadcast *);
    EXPORT virtual void visit(const Call *);
    EXPORT virtual void visit(const Let *);
    EXPORT virtual void visit(const LetStmt *);
    EXPORT virtual void visit(const AssertStmt *);
    EXPORT virtual void visit(const ProducerConsumer *);
    EXPORT virtual void visit(const For *);
    EXPORT virtual void visit(const Store *);
    EXPORT virtual void visit(const Provide *);
    EXPORT virtual void visit(const Allocate *);
    EXPORT virtual void visit(const Free *);
    EXPORT virtual void visit(const Realize *);
    EXPORT virtual void visit(const Block *);
    EXPORT virtual void visit(const IfThenElse *);
    EXPORT virtual void visit(const Evaluate *);
    EXPORT virtual void visit(const Shuffle *);
    EXPORT virtual void visit(const Prefetch *);
    // @}
};

}
}

#endif
#ifndef HALIDE_IR_MUTATOR_H
#define HALIDE_IR_MUTATOR_H

/** \file
 * Defines a base class for passes over the IR that modify it
 */


namespace Halide {
namespace Internal {

/** A base class for passes over the IR which modify it
 * (e.g. replacing a variable with a value (Substitute.h), or
 * constant-folding).
 *
 * Your mutate should override the visit methods you care about. Return
 * the new expression by assigning to expr or stmt. The default ones
 * recursively mutate their children. To mutate sub-expressions and
 * sub-statements you should the mutate method, which will dispatch to
 * the appropriate visit method and then return the value of expr or
 * stmt after the call to visit.
 */
class IRMutator : public IRVisitor {
public:
    EXPORT virtual ~IRMutator();

    /** This is the main interface for using a mutator. Also call
     * these in your subclass to mutate sub-expressions and
     * sub-statements.
     */
    EXPORT virtual Expr mutate(const Expr &expr);
    EXPORT virtual Stmt mutate(const Stmt &stmt);

protected:

    /** visit methods that take Exprs assign to this to return their
     * new value */
    Expr expr;

    /** visit methods that take Stmts assign to this to return their
     * new value */
    Stmt stmt;

    EXPORT virtual void visit(const IntImm *);
    EXPORT virtual void visit(const UIntImm *);
    EXPORT virtual void visit(const FloatImm *);
    EXPORT virtual void visit(const StringImm *);
    EXPORT virtual void visit(const Cast *);
    EXPORT virtual void visit(const Variable *);
    EXPORT virtual void visit(const Add *);
    EXPORT virtual void visit(const Sub *);
    EXPORT virtual void visit(const Mul *);
    EXPORT virtual void visit(const Div *);
    EXPORT virtual void visit(const Mod *);
    EXPORT virtual void visit(const Min *);
    EXPORT virtual void visit(const Max *);
    EXPORT virtual void visit(const EQ *);
    EXPORT virtual void visit(const NE *);
    EXPORT virtual void visit(const LT *);
    EXPORT virtual void visit(const LE *);
    EXPORT virtual void visit(const GT *);
    EXPORT virtual void visit(const GE *);
    EXPORT virtual void visit(const And *);
    EXPORT virtual void visit(const Or *);
    EXPORT virtual void visit(const Not *);
    EXPORT virtual void visit(const Select *);
    EXPORT virtual void visit(const Load *);
    EXPORT virtual void visit(const Ramp *);
    EXPORT virtual void visit(const Broadcast *);
    EXPORT virtual void visit(const Call *);
    EXPORT virtual void visit(const Let *);
    EXPORT virtual void visit(const LetStmt *);
    EXPORT virtual void visit(const AssertStmt *);
    EXPORT virtual void visit(const ProducerConsumer *);
    EXPORT virtual void visit(const For *);
    EXPORT virtual void visit(const Store *);
    EXPORT virtual void visit(const Provide *);
    EXPORT virtual void visit(const Allocate *);
    EXPORT virtual void visit(const Free *);
    EXPORT virtual void visit(const Realize *);
    EXPORT virtual void visit(const Block *);
    EXPORT virtual void visit(const IfThenElse *);
    EXPORT virtual void visit(const Evaluate *);
    EXPORT virtual void visit(const Shuffle *);
    EXPORT virtual void visit(const Prefetch *);
};


/** A mutator that caches and reapplies previously-done mutations, so
 * that it can handle graphs of IR that have not had CSE done to
 * them. */
class IRGraphMutator : public IRMutator {
protected:
    std::map<Expr, Expr, ExprCompare> expr_replacements;
    std::map<Stmt, Stmt, Stmt::Compare> stmt_replacements;

public:
    EXPORT Stmt mutate(const Stmt &s);
    EXPORT Expr mutate(const Expr &e);
};


}
}

#endif

namespace Halide {
namespace Internal {

typedef std::map<std::string, Interval> DimBounds;

const int64_t unknown = std::numeric_limits<int64_t>::min();

/** Visitor for keeping track of functions that are directly called and the
 * arguments with which they are called. */
class FindAllCalls : public IRVisitor {
    using IRVisitor::visit;

    void visit(const Call *call) {
        if (call->call_type == Call::Halide || call->call_type == Call::Image) {
            funcs_called.insert(call->name);
            call_args.push_back(std::make_pair(call->name, call->args));
        }
        for (size_t i = 0; i < call->args.size(); i++) {
            call->args[i].accept(this);
        }
    }
public:
    std::set<std::string> funcs_called;
    std::vector<std::pair<std::string, std::vector<Expr>>> call_args;
};

/** Return a map of estimates on some variables. */
class GetVarEstimates : public IRVisitor {
    using IRVisitor::visit;

    void visit(const Variable *var) {
        if (var->param.defined() && !var->param.is_buffer() &&
            var->param.get_estimate().defined()) {
            var_estimates[var->param.name()] = var->param.get_estimate();
        }
    }
public:
    std::map<std::string, Expr> var_estimates;
};


/** Substitute every variable with its estimate if specified. */
class SubstituteVarEstimates: public IRMutator {
    using IRMutator::visit;

    void visit(const Variable *var) {
        if (var->param.defined() && !var->param.is_buffer() &&
            var->param.get_estimate().defined()) {
            expr = var->param.get_estimate();
        } else {
            expr = var;
        }
    }
};

/** Return the size of an interval. Return an undefined expr if the interval
 * is unbounded. */
Expr get_extent(const Interval &i);

/** Return the size of an n-d box. */
Expr box_size(const Box &b);

/** Helper function to print the bounds of a region. */
void disp_regions(const std::map<std::string, Box> &regions);

/** Return the corresponding definition of a function given the stage. */
Definition get_stage_definition(const Function &f, int stage_num);

/** Add partial load costs to the corresponding function in the result costs. */
void combine_load_costs(std::map<std::string, Expr> &result,
                        const std::map<std::string, Expr> &partial);

/** Return the required bounds of an intermediate stage (f, stage_num) of
 * function 'f' given the bounds of the pure dimensions. */
DimBounds get_stage_bounds(Function f, int stage_num, const DimBounds &pure_bounds);

/** Return the required bounds for all the stages of the function 'f'. Each entry
 * in the returned vector corresponds to a stage. */
std::vector<DimBounds> get_stage_bounds(Function f, const DimBounds &pure_bounds);

/** Recursively inline all the functions in the set 'inlines' into the
 * expression 'e' and return the resulting expression. */
Expr perform_inline(Expr e, const std::map<std::string, Function> &env,
                    const std::set<std::string> &inlines = std::set<std::string>());

/** Return all functions that are directly called by a function stage (f, stage). */
std::set<std::string> get_parents(Function f, int stage);

/** Return value of element within a map. This will assert if the element is not
 * in the map. */
// @{
template<typename K, typename V>
V get_element(const std::map<K, V> &m, const K &key) {
    const auto &iter = m.find(key);
    internal_assert(iter != m.end());
    return iter->second;
}

template<typename K, typename V>
V &get_element(std::map<K, V> &m, const K &key) {
    const auto &iter = m.find(key);
    internal_assert(iter != m.end());
    return iter->second;
}
// @}

}
}

#endif
#ifndef HALIDE_BOUNDARY_CONDITIONS_H
#define HALIDE_BOUNDARY_CONDITIONS_H

/** \file
 * Support for imposing boundary conditions on Halide::Funcs.
 */

#include <utility>
#include <vector>

#ifndef HALIDE_FUNC_H
#define HALIDE_FUNC_H

/** \file
 *
 * Defines Func - the front-end handle on a halide function, and related classes.
 */

#ifndef HALIDE_VAR_H
#define HALIDE_VAR_H

/** \file
 * Defines the Var - the front-end variable
 */


namespace Halide {

/** A Halide variable, to be used when defining functions. It is just
 * a name, and can be reused in places where no name conflict will
 * occur. It can be used in the left-hand-side of a function
 * definition, or as an Expr. As an Expr, it always has type
 * Int(32). */
class Var {
    std::string _name;
public:
    /** Construct a Var with the given name */
    EXPORT Var(const std::string &n);

    /** Construct a Var with an automatically-generated unique name. */
    EXPORT Var();

    /** Get the name of a Var */
    const std::string &name() const {return _name;}

    /** Test if two Vars are the same. This simply compares the names. */
    bool same_as(const Var &other) const {return _name == other._name;}

    /** Implicit var constructor. Implicit variables are injected
     * automatically into a function call if the number of arguments
     * to the function are fewer than its dimensionality and a
     * placeholder ("_") appears in its argument list. Defining a
     * function to equal an expression containing implicit variables
     * similarly appends those implicit variables, in the same order,
     * to the left-hand-side of the definition where the placeholder
     * ('_') appears.
     *
     * For example, consider the definition:
     *
     \code
     Func f, g;
     Var x, y;
     f(x, y) = 3;
     \endcode
     *
     * A call to f with the placeholder symbol \ref _
     * will have implicit arguments injected automatically, so f(2, \ref _)
     * is equivalent to f(2, \ref _0), where \ref _0 = Var::implicit(0), and f(\ref _)
     * (and indeed f when cast to an Expr) is equivalent to f(\ref _0, \ref _1).
     * The following definitions are all equivalent, differing only in the
     * variable names.
     *
     \code
     g(_) = f*3;
     g(_) = f(_)*3;
     g(x, _) = f(x, _)*3;
     g(x, y) = f(x, y)*3;
     \endcode
     *
     * These are expanded internally as follows:
     *
     \code
     g(_0, _1) = f(_0, _1)*3;
     g(_0, _1) = f(_0, _1)*3;
     g(x, _0) = f(x, _0)*3;
     g(x, y) = f(x, y)*3;
     \endcode
     *
     * The following, however, defines g as four dimensional:
     \code
     g(x, y, _) = f*3;
     \endcode
     *
     * It is equivalent to:
     *
     \code
     g(x, y, _0, _1) = f(_0, _1)*3;
     \endcode
     *
     * Expressions requiring differing numbers of implicit variables
     * can be combined. The left-hand-side of a definition injects
     * enough implicit variables to cover all of them:
     *
     \code
     Func h;
     h(x) = x*3;
     g(x) = h + (f + f(x)) * f(x, y);
     \endcode
     *
     * expands to:
     *
     \code
     Func h;
     h(x) = x*3;
     g(x, _0, _1) = h(_0) + (f(_0, _1) + f(x, _0)) * f(x, y);
     \endcode
     *
     * The first ten implicits, _0 through _9, are predeclared in this
     * header and can be used for scheduling. They should never be
     * used as arguments in a declaration or used in a call.
     *
     * While it is possible to use Var::implicit or the predeclared
     * implicits to create expressions that can be treated as small
     * anonymous functions (e.g. Func(_0 + _1)) this is considered
     * poor style. Instead use \ref lambda.
     */
    EXPORT static Var implicit(int n);

    /** Return whether a variable name is of the form for an implicit argument.
     * TODO: This is almost guaranteed to incorrectly fire on user
     * declared variables at some point. We should likely prevent
     * user Var declarations from making names of this form.
     */
    //{
    EXPORT static bool is_implicit(const std::string &name);
    bool is_implicit() const {
        return is_implicit(name());
    }
    //}

    /** Return the argument index for a placeholder argument given its
     *  name. Returns 0 for \ref _0, 1 for \ref _1, etc. Returns -1 if
     *  the variable is not of implicit form.
     */
    //{
    static int implicit_index(const std::string &name) {
        return is_implicit(name) ? atoi(name.c_str() + 1) : -1;
    }
    int implicit_index() const {
        return implicit_index(name());
    }
    //}

    /** Test if a var is the placeholder variable \ref _ */
    //{
    static bool is_placeholder(const std::string &name) {
        return name == "_";
    }
    bool is_placeholder() const {
        return is_placeholder(name());
    }
    //}

    /** A Var can be treated as an Expr of type Int(32) */
    operator Expr() const {
        return Internal::Variable::make(Int(32), name());
    }

    /** Vars to use for scheduling producer/consumer pairs on the gpu. Deprecated. */
    // @{
    HALIDE_ATTRIBUTE_DEPRECATED("Var::gpu_blocks() is deprecated.")
    static Var gpu_blocks() {
        return Var("__deprecated_block_id_x");
    }
    HALIDE_ATTRIBUTE_DEPRECATED("Var::gpu_threads() is deprecated.")
    static Var gpu_threads() {
        return Var("__deprecated_thread_id_x");
    }
    // @}

    /** A Var that represents the location outside the outermost loop. */
    static Var outermost() {
        return Var("__outermost");
    }

};

/** A placeholder variable for infered arguments. See \ref Var::implicit */
EXPORT extern Var _;

/** The first ten implicit Vars for use in scheduling. See \ref Var::implicit */
// @{
EXPORT extern Var _0, _1, _2, _3, _4, _5, _6, _7, _8, _9;
// @}

}

#endif
#ifndef HALIDE_PARAM_H
#define HALIDE_PARAM_H

#include <type_traits>


/** \file
 *
 * Classes for declaring scalar parameters to halide pipelines
 */

namespace Halide {

/** A scalar parameter to a halide pipeline. If you're jitting, this
 * should be bound to an actual value of type T using the set method
 * before you realize the function uses this. If you're statically
 * compiling, this param should appear in the argument list. */
template<typename T>
class Param {
    /** A reference-counted handle on the internal parameter object */
    Internal::Parameter param;

    void check_name() const {
        user_assert(param.name() != "__user_context") << "Param<void*>(\"__user_context\") "
            << "is no longer used to control whether Halide functions take explicit "
            << "user_context arguments. Use set_custom_user_context() when jitting, "
            << "or add Target::UserContext to the Target feature set when compiling ahead of time.";
    }

public:
    /** Construct a scalar parameter of type T with a unique
     * auto-generated name */
    Param() :
        param(type_of<T>(), false, 0, Internal::make_entity_name(this, "Halide::Param<?", 'p')) {}

    /** Construct a scalar parameter of type T with the given name. */
    // @{
    explicit Param(const std::string &n) :
        param(type_of<T>(), false, 0, n, /*is_explicit_name*/ true) {
        check_name();
    }
    explicit Param(const char *n) :
        param(type_of<T>(), false, 0, n, /*is_explicit_name*/ true) {
        check_name();
    }
    // @}

    /** Construct a scalar parameter of type T an initial value of
     * 'val'. Only triggers for non-pointer types. */
    template <typename T2 = T, typename std::enable_if<!std::is_pointer<T2>::value>::type * = nullptr>
    explicit Param(T val) :
        param(type_of<T>(), false, 0, Internal::make_entity_name(this, "Halide::Param<?", 'p')) {
        set(val);
    }

    /** Construct a scalar parameter of type T with the given name
     * and an initial value of 'val'. */
    Param(const std::string &n, T val) :
        param(type_of<T>(), false, 0, n, /*is_explicit_name*/ true) {
        check_name();
        set(val);
    }

    /** Construct a scalar parameter of type T with an initial value of 'val'
    * and a given min and max. */
    Param(T val, Expr min, Expr max) :
        param(type_of<T>(), false, 0, Internal::make_entity_name(this, "Halide::Param<?", 'p')) {
        set_range(min, max);
        set(val);
    }

    /** Construct a scalar parameter of type T with the given name
     * and an initial value of 'val' and a given min and max. */
    Param(const std::string &n, T val, Expr min, Expr max) :
        param(type_of<T>(), false, 0, n, /*is_explicit_name*/ true) {
        check_name();
        set_range(min, max);
        set(val);
    }

    /** Get the name of this parameter */
    const std::string &name() const {
        return param.name();
    }

    /** Return true iff the name was explicitly specified in the ctor (vs autogenerated). */
    bool is_explicit_name() const {
        return param.is_explicit_name();
    }

    /** Get the current value of this parameter. Only meaningful when jitting. */
    NO_INLINE T get() const {
        return param.get_scalar<T>();
    }

    /** Set the current value of this parameter. Only meaningful when jitting */
    NO_INLINE void set(T val) {
        param.set_scalar<T>(val);
    }

    /** Get a pointer to the location that stores the current value of
     * this parameter. Only meaningful for jitting. */
    NO_INLINE T *get_address() const {
        return (T *)(param.get_scalar_address());
    }

    /** Get the halide type of T */
    Type type() const {
        return type_of<T>();
    }

    /** Get or set the possible range of this parameter. Use undefined
     * Exprs to mean unbounded. */
    // @{
    void set_range(Expr min, Expr max) {
        set_min_value(min);
        set_max_value(max);
    }

    void set_min_value(Expr min) {
        if (min.defined() && min.type() != type_of<T>()) {
            min = Internal::Cast::make(type_of<T>(), min);
        }
        param.set_min_value(min);
    }

    void set_max_value(Expr max) {
        if (max.defined() && max.type() != type_of<T>()) {
            max = Internal::Cast::make(type_of<T>(), max);
        }
        param.set_max_value(max);
    }

    Expr get_min_value() const {
        return param.get_min_value();
    }

    Expr get_max_value() const {
        return param.get_max_value();
    }
    // @}

    void set_estimate(const T &value) {
        param.set_estimate(Expr(value));
    }

    /** You can use this parameter as an expression in a halide
     * function definition */
    operator Expr() const {
        return Internal::Variable::make(type_of<T>(), name(), param);
    }

    /** Using a param as the argument to an external stage treats it
     * as an Expr */
    operator ExternFuncArgument() const {
        return Expr(*this);
    }

    /** Construct the appropriate argument matching this parameter,
     * for the purpose of generating the right type signature when
     * statically compiling halide pipelines. */
    operator Argument() const {
        return Argument(name(), Argument::InputScalar, type(), 0,
            param.get_scalar_expr(), param.get_min_value(), param.get_max_value());
    }
};

/** Returns an Expr corresponding to the user context passed to
 * the function (if any). It is rare that this function is necessary
 * (e.g. to pass the user context to an extern function written in C). */
inline Expr user_context_value() {
    return Internal::Variable::make(Handle(), "__user_context",
        Internal::Parameter(Handle(), false, 0, "__user_context", true));
}

}  // namespace Halide

#endif
#ifndef HALIDE_OUTPUT_IMAGE_PARAM_H
#define HALIDE_OUTPUT_IMAGE_PARAM_H

/** \file
 *
 * Classes for declaring output image parameters to halide pipelines
 */


namespace Halide {

/** A handle on the output buffer of a pipeline. Used to make static
 * promises about the output size and stride. */
class OutputImageParam {
protected:
    /** A reference-counted handle on the internal parameter object */
    Internal::Parameter param;

    /** Is this an input or an output? OutputImageParam is the base class for both. */
    Argument::Kind kind;

    void add_implicit_args_if_placeholder(std::vector<Expr> &args,
                                          Expr last_arg,
                                          int total_args,
                                          bool *placeholder_seen) const;
public:

    /** Construct a null image parameter handle. */
    OutputImageParam() {}

    /** Construct an OutputImageParam that wraps an Internal Parameter object. */
    EXPORT OutputImageParam(const Internal::Parameter &p, Argument::Kind k);

    /** Get the name of this Param */
    EXPORT const std::string &name() const;

    /** Get the type of the image data this Param refers to */
    EXPORT Type type() const;

    /** Is this parameter handle non-nullptr */
    EXPORT bool defined() const;

    /** Get a handle on one of the dimensions for the purposes of
     * inspecting or constraining its min, extent, or stride. */
    EXPORT Internal::Dimension dim(int i);

    /** Get a handle on one of the dimensions for the purposes of
     * inspecting its min, extent, or stride. */
    EXPORT const Internal::Dimension dim(int i) const;

    /** Get the alignment of the host pointer in bytes. Defaults to
     * the size of type. */
    EXPORT int host_alignment() const;

    /** Set the expected alignment of the host pointer in bytes. */
    EXPORT OutputImageParam &set_host_alignment(int);

    /** Get the dimensionality of this image parameter */
    EXPORT int dimensions() const;

    /** Get an expression giving the minimum coordinate in dimension 0, which
     * by convention is the coordinate of the left edge of the image */
    EXPORT Expr left() const;

    /** Get an expression giving the maximum coordinate in dimension 0, which
     * by convention is the coordinate of the right edge of the image */
    EXPORT Expr right() const;

    /** Get an expression giving the minimum coordinate in dimension 1, which
     * by convention is the top of the image */
    EXPORT Expr top() const;

    /** Get an expression giving the maximum coordinate in dimension 1, which
     * by convention is the bottom of the image */
    EXPORT Expr bottom() const;

    /** Get an expression giving the extent in dimension 0, which by
     * convention is the width of the image */
    EXPORT Expr width() const;

    /** Get an expression giving the extent in dimension 1, which by
     * convention is the height of the image */
    EXPORT Expr height() const;

    /** Get an expression giving the extent in dimension 2, which by
     * convention is the channel-count of the image */
    EXPORT Expr channels() const;

    /** Get at the internal parameter object representing this ImageParam. */
    EXPORT Internal::Parameter parameter() const;

    /** Construct the appropriate argument matching this parameter,
     * for the purpose of generating the right type signature when
     * statically compiling halide pipelines. */
    EXPORT operator Argument() const;

    /** Using a param as the argument to an external stage treats it
     * as an Expr */
    EXPORT operator ExternFuncArgument() const;
};

}

#endif
#ifndef HALIDE_RDOM_H
#define HALIDE_RDOM_H

/** \file
 * Defines the front-end syntax for reduction domains and reduction
 * variables.
 */


#include <vector>

namespace Halide {

class ImageParam;

/** A reduction variable represents a single dimension of a reduction
 * domain (RDom). Don't construct them directly, instead construct an
 * RDom, and use RDom::operator[] to get at the variables. For
 * single-dimensional reduction domains, you can just cast a
 * single-dimensional RDom to an RVar. */
class RVar {
    std::string _name;
    Internal::ReductionDomain _domain;
    int _index;

    const Internal::ReductionVariable &_var() const {
        return _domain.domain().at(_index);
    }

public:
    /** An empty reduction variable. */
    RVar() : _name(Internal::make_entity_name(this, "Halide::RVar", 'r')) {}

    /** Construct an RVar with the given name */
    explicit RVar(const std::string &n) : _name(n) {
    }

    /** Construct a reduction variable with the given name and
     * bounds. Must be a member of the given reduction domain. */
    RVar(Internal::ReductionDomain domain, int index) :
        _domain(domain), _index(index) {
    }

    /** The minimum value that this variable will take on */
    EXPORT Expr min() const;

    /** The number that this variable will take on. The maximum value
     * of this variable will be min() + extent() - 1 */
    EXPORT Expr extent() const;

    /** The reduction domain this is associated with. */
    EXPORT Internal::ReductionDomain domain() const {return _domain;}

    /** The name of this reduction variable */
    EXPORT const std::string &name() const;

    /** Reduction variables can be used as expressions. */
    EXPORT operator Expr() const;
};

/** A multi-dimensional domain over which to iterate. Used when
 * defining functions with update definitions.
 *
 * An reduction is a function with a two-part definition. It has an
 * initial value, which looks much like a pure function, and an update
 * definition, which may refer to some RDom. Evaluating such a
 * function first initializes it over the required domain (which is
 * inferred based on usage), and then runs update rule for all points
 * in the RDom. For example:
 *
 \code
 Func f;
 Var x;
 RDom r(0, 10);
 f(x) = x; // the initial value
 f(r) = f(r) * 2;
 Buffer<int> result = f.realize(10);
 \endcode
 *
 * This function creates a single-dimensional buffer of size 10, in
 * which element x contains the value x*2. Internally, first the
 * initialization rule fills in x at every site, and then the update
 * definition doubles every site.
 *
 * One use of reductions is to build a function recursively (pure
 * functions in halide cannot be recursive). For example, this
 * function fills in an array with the first 20 fibonacci numbers:
 *
 \code
 Func f;
 Var x;
 RDom r(2, 18);
 f(x) = 1;
 f(r) = f(r-1) + f(r-2);
 \endcode
 *
 * Another use of reductions is to perform scattering operations, as
 * unlike a pure function declaration, the left-hand-side of an update
 * definition may contain general expressions:
 *
 \code
 ImageParam input(UInt(8), 2);
 Func histogram;
 Var x;
 RDom r(input); // Iterate over all pixels in the input
 histogram(x) = 0;
 histogram(input(r.x, r.y)) = histogram(input(r.x, r.y)) + 1;
 \endcode
 *
 * An update definition may also be multi-dimensional. This example
 * computes a summed-area table by first summing horizontally and then
 * vertically:
 *
 \code
 ImageParam input(Float(32), 2);
 Func sum_x, sum_y;
 Var x, y;
 RDom r(input);
 sum_x(x, y)     = input(x, y);
 sum_x(r.x, r.y) = sum_x(r.x, r.y) + sum_x(r.x-1, r.y);
 sum_y(x, y)     = sum_x(x, y);
 sum_y(r.x, r.y) = sum_y(r.x, r.y) + sum_y(r.x, r.y-1);
 \endcode
 *
 * You can also mix pure dimensions with reduction variables. In the
 * previous example, note that there's no need for the y coordinate in
 * sum_x to be traversed serially. The sum within each row is entirely
 * independent. The rows could be computed in parallel, or in a
 * different order, without changing the meaning. Therefore, we can
 * instead write this definition as follows:
 *
 \code
 ImageParam input(Float(32), 2);
 Func sum_x, sum_y;
 Var x, y;
 RDom r(input);
 sum_x(x, y)   = input(x, y);
 sum_x(r.x, y) = sum_x(r.x, y) + sum_x(r.x-1, y);
 sum_y(x, y)   = sum_x(x, y);
 sum_y(x, r.y) = sum_y(x, r.y) + sum_y(x, r.y-1);
 \endcode
 *
 * This lets us schedule it more flexibly. You can now parallelize the
 * update step of sum_x over y by calling:
 \code
 sum_x.update().parallel(y).
 \endcode
 *
 * Note that calling sum_x.parallel(y) only parallelizes the
 * initialization step, and not the update step! Scheduling the update
 * step of a reduction must be done using the handle returned by
 * \ref Func::update(). This code parallelizes both the initialization
 * step and the update step:
 *
 \code
 sum_x.parallel(y);
 sum_x.update().parallel(y);
 \endcode
 *
 * When you mix reduction variables and pure dimensions, the reduction
 * domain is traversed outermost. That is, for each point in the
 * reduction domain, the inferred pure domain is traversed in its
 * entirety. For the above example, this means that sum_x walks down
 * the columns, and sum_y walks along the rows. This may not be
 * cache-coherent. You may try reordering these dimensions using the
 * schedule, but Halide will return an error if it decides that this
 * risks changing the meaning of your function. The solution lies in
 * clever scheduling. If we say:
 *
 \code
 sum_x.compute_at(sum_y, y);
 \endcode
 *
 * Then the sum in x is computed only as necessary for each scanline
 * of the sum in y. This not only results in sum_x walking along the
 * rows, it also improves the locality of the entire pipeline.
 */
class RDom {
    Internal::ReductionDomain dom;

    void init_vars(const std::string &name);

    EXPORT void initialize_from_ranges(const std::vector<std::pair<Expr, Expr>> &ranges, std::string name = "");

    template <typename... Args>
    NO_INLINE void initialize_from_ranges(std::vector<std::pair<Expr, Expr>> &ranges, Expr min, Expr extent, Args&&... args) {
        ranges.push_back({ min, extent });
        initialize_from_ranges(ranges, std::forward<Args>(args)...);
    }

public:
    /** Construct an undefined reduction domain. */
    EXPORT RDom() {}

    /** Construct a multi-dimensional reduction domain with the given name. If the name
     * is left blank, a unique one is auto-generated. */
    // @{
    NO_INLINE RDom(const std::vector<std::pair<Expr, Expr>> &ranges, std::string name = "") {
        initialize_from_ranges(ranges, name);
    }

    template <typename... Args>
    NO_INLINE RDom(Expr min, Expr extent, Args&&... args) {
        // This should really just be a delegating constructor, but I couldn't make
        // that work with variadic template unpacking in visual studio 2013
        std::vector<std::pair<Expr, Expr>> ranges;
        initialize_from_ranges(ranges, min, extent, std::forward<Args>(args)...);
    }
    // @}

    /** Construct a reduction domain that iterates over all points in
     * a given Buffer or ImageParam. Has the same dimensionality as
     * the argument. */
    // @{
    EXPORT RDom(const Buffer<> &);
    EXPORT RDom(ImageParam);
    EXPORT RDom(const Halide::Internal::Constrainable &);  // Allows Input<Buffer<>>
    template<typename T>
    NO_INLINE RDom(const Buffer<T> &im) : RDom(Buffer<>(im)) {}
    // @}

    /** Construct a reduction domain that wraps an Internal ReductionDomain object. */
    EXPORT RDom(Internal::ReductionDomain d);

    /** Get at the internal reduction domain object that this wraps. */
    Internal::ReductionDomain domain() const {return dom;}

    /** Check if this reduction domain is non-null */
    bool defined() const {return dom.defined();}

    /** Compare two reduction domains for equality of reference */
    bool same_as(const RDom &other) const {return dom.same_as(other.dom);}

    /** Get the dimensionality of a reduction domain */
    EXPORT int dimensions() const;

    /** Get at one of the dimensions of the reduction domain */
    EXPORT RVar operator[](int) const;

    /** Single-dimensional reduction domains can be used as RVars directly. */
    EXPORT operator RVar() const;

    /** Single-dimensional reduction domains can be also be used as Exprs directly. */
    EXPORT operator Expr() const;

    /** Add a predicate to the RDom. An RDom may have multiple
     * predicates associated with it. An update definition that uses
     * an RDom only iterates over the subset points in the domain for
     * which all of its predicates are true. The predicate expression
     * obeys the same rules as the expressions used on the
     * right-hand-side of the corresponding update definition. It may
     * refer to the RDom's variables and free variables in the Func's
     * update definition. It may include calls to other Funcs, or make
     * recursive calls to the same Func. This permits iteration over
     * non-rectangular domains, or domains with sizes that vary with
     * some free variable, or domains with shapes determined by some
     * other Func.
     *
     * Note that once RDom is used in the update definition of some
     * Func, no new predicates can be added to the RDom.
     *
     * Consider a simple example:
     \code
     RDom r(0, 20, 0, 20);
     r.where(r.x < r.y);
     r.where(r.x == 10);
     r.where(r.y > 13);
     f(r.x, r.y) += 1;
     \endcode
     * This is equivalent to:
     \code
     for (int r.y = 0; r.y < 20; r.y++) {
       if (r.y > 13) {
         for (int r.x = 0; r.x < 20; r.x++) {
           if (r.x == 10) {
             if (r.x < r.y) {
               f[r.x, r.y] += 1;
             }
           }
         }
       }
     }
     \endcode
     *
     * Where possible Halide restricts the range of the containing for
     * loops to avoid the cases where the predicate is false so that
     * the if statement can be removed entirely. The case above would
     * be further simplified into:
     *
     \code
     for (int r.y = 14; r.y < 20; r.y++) {
       f[r.x, r.y] += 1;
     }
     \endcode
     *
     * In general, the predicates that we can simplify away by
     * restricting loop ranges are inequalities that compare an inner
     * Var or RVar to some expression in outer Vars or RVars.
     *
     * You can also pack multiple conditions into one predicate like so:
     *
     \code
     RDom r(0, 20, 0, 20);
     r.where((r.x < r.y) && (r.x == 10) && (r.y > 13));
     f(r.x, r.y) += 1;
     \endcode
     *
     */
    EXPORT void where(Expr predicate);

    /** Direct access to the first four dimensions of the reduction
     * domain. Some of these variables may be undefined if the
     * reduction domain has fewer than four dimensions. */
    // @{
    RVar x, y, z, w;
    // @}
};

/** Emit an RVar in a human-readable form */
std::ostream &operator<<(std::ostream &stream, RVar);

/** Emit an RDom in a human-readable form. */
std::ostream &operator<<(std::ostream &stream, RDom);
}

#endif
#ifndef HALIDE_JIT_MODULE_H
#define HALIDE_JIT_MODULE_H

/** \file
 * Defines the struct representing lifetime and dependencies of
 * a JIT compiled halide pipeline
 */

#include <map>
#include <memory>


namespace llvm {
class Module;
class Type;
}

namespace Halide {

struct ExternCFunction;
struct JITExtern;
struct Target;
class Module;

namespace Internal {

class JITModuleContents;
struct LoweredFunc;

struct JITModule {
    IntrusivePtr<JITModuleContents> jit_module;

    struct Symbol {
        void *address;
        llvm::Type *llvm_type;
        Symbol() : address(nullptr), llvm_type(nullptr) {}
        Symbol(void *address, llvm::Type *llvm_type) : address(address), llvm_type(llvm_type) {}
    };

    EXPORT JITModule();
    EXPORT JITModule(const Module &m, const LoweredFunc &fn,
                     const std::vector<JITModule> &dependencies = std::vector<JITModule>());
    /** The exports map of a JITModule contains all symbols which are
     * available to other JITModules which depend on this one. For
     * runtime modules, this is all of the symbols exported from the
     * runtime. For a JITted Func, it generally only contains the main
     * result Func of the compilation, which takes its name directly
     * from the Func declaration. One can also make a module which
     * contains no code itself but is just an exports maps providing
     * arbitrary pointers to functions or global variables to JITted
     * code. */
    EXPORT const std::map<std::string, Symbol> &exports() const;

    /** A pointer to the raw halide function. Its true type depends
     * on the Argument vector passed to CodeGen_LLVM::compile. Image
     * parameters become (halide_buffer_t *), and scalar parameters become
     * pointers to the appropriate values. The final argument is a
     * pointer to the halide_buffer_t defining the output. This will be nullptr for
     * a JITModule which has not yet been compiled or one that is not
     * a Halide Func compilation at all. */
    EXPORT void *main_function() const;

    /** Returns the Symbol structure for the routine documented in
     * main_function. Returning a Symbol allows access to the LLVM
     * type as well as the address. The address and type will be nullptr
     * if the module has not been compiled. */
    EXPORT Symbol entrypoint_symbol() const;

    /** Returns the Symbol structure for the argv wrapper routine
     * corresponding to the entrypoint. The argv wrapper is callable
     * via an array of void * pointers to the arguments for the
     * call. Returning a Symbol allows access to the LLVM type as well
     * as the address. The address and type will be nullptr if the module
     * has not been compiled. */
    EXPORT Symbol argv_entrypoint_symbol() const;

    /** A slightly more type-safe wrapper around the raw halide
     * module. Takes it arguments as an array of pointers that
     * correspond to the arguments to \ref main_function . This will
     * be nullptr for a JITModule which has not yet been compiled or one
     * that is not a Halide Func compilation at all. */
    // @{
    typedef int (*argv_wrapper)(const void **args);
    EXPORT argv_wrapper argv_function() const;
    // @}

    /** Add another JITModule to the dependency chain. Dependencies
     * are searched to resolve symbols not found in the current
     * compilation unit while JITting. */
    EXPORT void add_dependency(JITModule &dep);
    /** Registers a single Symbol as available to modules which depend
     * on this one. The Symbol structure provides both the address and
     * the LLVM type for the function, which allows type safe linkage of
     * extenal routines. */
    EXPORT void add_symbol_for_export(const std::string &name, const Symbol &extern_symbol);
    /** Registers a single function as available to modules which
     * depend on this one. This routine converts the ExternSignature
     * info into an LLVM type, which allows type safe linkage of
     * external routines. */
    EXPORT void add_extern_for_export(const std::string &name,
                                      const ExternCFunction &extern_c_function);

    /** Look up a symbol by name in this module or its dependencies. */
    EXPORT Symbol find_symbol_by_name(const std::string &) const;

    /** Take an llvm module and compile it. The requested exports will
        be available via the exports method. */
    EXPORT void compile_module(std::unique_ptr<llvm::Module> mod,
                               const std::string &function_name, const Target &target,
                               const std::vector<JITModule> &dependencies = std::vector<JITModule>(),
                               const std::vector<std::string> &requested_exports = std::vector<std::string>());

    /** Encapsulate device (GPU) and buffer interactions. */
    EXPORT void memoization_cache_set_size(int64_t size) const;

    /** Return true if compile_module has been called on this module. */
    EXPORT bool compiled() const;
};

typedef int (*halide_task)(void *user_context, int, uint8_t *);

struct JITHandlers {
    void (*custom_print)(void *, const char *){nullptr};
    void *(*custom_malloc)(void *, size_t){nullptr};
    void (*custom_free)(void *, void *){nullptr};
    int (*custom_do_task)(void *, halide_task, int, uint8_t *){nullptr};
    int (*custom_do_par_for)(void *, halide_task, int, int, uint8_t *){nullptr};
    void (*custom_error)(void *, const char *){nullptr};
    int32_t (*custom_trace)(void *, const halide_trace_event_t *){nullptr};
    void *(*custom_get_symbol)(const char *name){nullptr};
    void *(*custom_load_library)(const char *name){nullptr};
    void *(*custom_get_library_symbol)(void *lib, const char *name){nullptr};
};

struct JITUserContext {
    void *user_context;
    JITHandlers handlers;
};

class JITSharedRuntime {
public:
    // Note only the first llvm::Module passed in here is used. The same shared runtime is used for all JIT.
    EXPORT static std::vector<JITModule> get(llvm::Module *m, const Target &target, bool create = true);
    EXPORT static void init_jit_user_context(JITUserContext &jit_user_context, void *user_context, const JITHandlers &handlers);
    EXPORT static JITHandlers set_default_handlers(const JITHandlers &handlers);

    /** Set the maximum number of bytes used by memoization caching.
     * If you are compiling statically, you should include HalideRuntime.h
     * and call halide_memoization_cache_set_size() instead.
     */
    EXPORT static void memoization_cache_set_size(int64_t size);

    EXPORT static void release_all();
};

}
}

#endif
#ifndef HALIDE_TUPLE_H
#define HALIDE_TUPLE_H

/** \file
 *
 * Defines Tuple - the front-end handle on small arrays of expressions.
 */


namespace Halide {

class FuncRef;

/** Create a small array of Exprs for defining and calling functions
 * with multiple outputs. */
class Tuple {
private:
    std::vector<Expr> exprs;
public:
    /** The number of elements in the tuple. */
    size_t size() const { return exprs.size(); }

    /** Get a reference to an element. */
    Expr &operator[](size_t x) {
        user_assert(x < exprs.size()) << "Tuple access out of bounds\n";
        return exprs[x];
    }

    /** Get a copy of an element. */
    Expr operator[](size_t x) const {
        user_assert(x < exprs.size()) << "Tuple access out of bounds\n";
        return exprs[x];
    }

    /** Construct a Tuple of a single Expr */
    explicit Tuple(Expr e) {
        exprs.push_back(e);
    }

    /** Construct a Tuple from some Exprs. */
    //@{
    template<typename ...Args>
    Tuple(Expr a, Expr b, Args&&... args) {
        exprs = std::vector<Expr>{a, b, std::forward<Args>(args)...};
    }
    //@}

    /** Construct a Tuple from a vector of Exprs */
    explicit NO_INLINE Tuple(const std::vector<Expr> &e) : exprs(e) {
        user_assert(e.size() > 0) << "Tuples must have at least one element\n";
    }

    /** Construct a Tuple from a function reference. */
    EXPORT Tuple(const FuncRef &);

    /** Treat the tuple as a vector of Exprs */
    const std::vector<Expr> &as_vector() const {
        return exprs;
    }
};

/** A Realization is a vector of references to existing Buffer
objects. Funcs with Tuple values return multiple images when you
realize them, and they return them as a Realization. Tuples are to
Exprs as Realizations are to Buffers. */
class Realization {
private:
    std::vector<Buffer<>> images;
public:
    /** The number of images in the Realization. */
    size_t size() const { return images.size(); }

    /** Get a const reference to one of the images. */
    const Buffer<> &operator[](size_t x) const {
        user_assert(x < images.size()) << "Realization access out of bounds\n";
        return images[x];
    }

    /** Get a reference to one of the images. */
    Buffer<> &operator[](size_t x) {
        user_assert(x < images.size()) << "Realization access out of bounds\n";
        return images[x];
    }

    /** Single-element realizations are implicitly castable to Buffers. */
    template<typename T>
    operator Buffer<T>() const {
        return images[0];
    }

    /** Construct a Realization that acts as a reference to some
     * existing Buffers. The element type of the Buffers may not be
     * const. */
    template<typename T,
             typename ...Args,
             typename = typename std::enable_if<Internal::all_are_convertible<Buffer<>, Args...>::value>::type>
    Realization(Buffer<T> &a, Args&&... args) {
        images = std::vector<Buffer<>>({a, args...});
    }

    /** Construct a Realization that refers to the buffers in an
     * existing vector of Buffer<> */
    explicit Realization(std::vector<Buffer<>> &e) : images(e) {
        user_assert(e.size() > 0) << "Realizations must have at least one element\n";
    }

    /** Call device_sync() for all Buffers in the Realization.
     * If one of the calls returns an error, subsequent Buffers won't have
     * device_sync called; thus callers should consider a nonzero return
     * code to mean that potentially all of the Buffers are in an indeterminate
     * state of sync.
     * Calling this explicitly should rarely be necessary, except for profiling. */
    int device_sync(void *ctx = nullptr) {
        for (auto &b : images) {
            int result = b.device_sync(ctx);
            if (result != 0) {
                return result;
            }
        }
        return 0;
    }

};

/** Equivalents of some standard operators for tuples. */
// @{
inline Tuple tuple_select(Tuple condition, const Tuple &true_value, const Tuple &false_value) {
    Tuple result(std::vector<Expr>(condition.size()));
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = select(condition[i], true_value[i], false_value[i]);
    }
    return result;
}

inline Tuple tuple_select(Expr condition, const Tuple &true_value, const Tuple &false_value) {
    Tuple result(std::vector<Expr>(true_value.size()));
    for (size_t i = 0; i < result.size(); i++) {
        result[i] = select(condition, true_value[i], false_value[i]);
    }
    return result;
}
// @}

}

#endif
#ifndef HALIDE_MODULE_H
#define HALIDE_MODULE_H

/** \file
 *
 * Defines Module, an IR container that fully describes a Halide program.
 */

#include <functional>

#ifndef HALIDE_EXTERNAL_CODE_H
#define HALIDE_EXTERNAL_CODE_H

#include <vector>


namespace Halide {

class ExternalCode {
private:
    enum Kind {
        LLVMBitcode,
        DeviceCode,
        CPlusPlusSource,
    } kind;

    Target llvm_target; // For LLVMBitcode.
    DeviceAPI device_code_kind;

    std::vector<uint8_t> code;

    // Used for debugging and naming the module to llvm.
    std::string nametag;

    ExternalCode(Kind kind, const Target &llvm_target, DeviceAPI device_api, const std::vector<uint8_t> &code, const std::string &name)
        : kind(kind), llvm_target(llvm_target), device_code_kind(device_api), code(code), nametag(name) {
    }

public:

    /** Construct an ExternalCode container from llvm bitcode. The
     * result can be passed to Halide::Module::append to have the
     * contained bitcode linked with that module. The Module's target
     * must match the target argument here on architecture, bit width,
     * and operating system. The name is used as a unique identifier
     * for the external code and duplicates will be reduced to a
     * single instance. Halide does not do anything other than to
     * compare names for equality. To guarantee uniqueness in public
     * code, we suggest using a Java style inverted domain name
     * followed by organization specific naming. E.g.:
     *     com.initech.y2k.5d2ac80aaf522eec6cb4b40f39fb923f9902bc7e */
    static ExternalCode bitcode_wrapper(const Target &cpu_type, const std::vector<uint8_t> &code, const std::string &name) {
        return ExternalCode(LLVMBitcode, cpu_type, DeviceAPI::None, code, name);
    }

    /** Construct an ExternalCode container from GPU "source code."
     * This container can be used to insert its code into the GPU code
     * generated for a given DeviceAPI. The specific type of code
     * depends on the device API used as follows:
     *     CUDA: llvm bitcode for PTX
     *     OpenCL: OpenCL source code
     *     GLSL: GLSL source code
     *     OpenGLCompute: GLSL source code
     *     Metal: Metal source code
     *     Hexagon: llvm bitcode for Hexagon
     *
     * At present, this API is not fully working. See Issue:
     *     https://github.com/halide/Halide/issues/1971
     *
     * The name is used as a unique identifier for the external code
     * and duplicates will be reduced to a single instance. Halide
     * does not do anything other than to compare names for
     * equality. To guarantee uniqueness in public code, we suggest
     * using a Java style inverted domain name followed by
     * organization specific naming. E.g.:
     *     com.tyrell.nexus-6.53947db86ba97a9ca5ecd5e60052880945bfeb37 */
    static ExternalCode device_code_wrapper(DeviceAPI device_api, const std::vector<uint8_t> &code, const std::string &name) {
        return ExternalCode(DeviceCode, Target(), device_api, code, name);
    }

    /** Construct an ExternalCode container from C++ source code. This
     * container can be used to insert its code into C++ output from
     * Halide.
     *
     * At present, this API is not fully working. See Issue:
     *     https://github.com/halide/Halide/issues/1971
     *
     * The name is used as a unique identifier for the external code
     * and duplicates will be reduced to a single instance. Halide
     * does not do anything other than to compare names for
     * equality. To guarantee uniqueness in public code, we suggest
     * using a Java style inverted domain name followed by
     * organization specific naming. E.g.:
     *     com.cyberdyne.skynet.78ad6c411d313f050f172cd3d440f23fdd797d0d */
    static ExternalCode c_plus_plus_code_wrapper(const std::vector<uint8_t> &code, const std::string &name) {
        return ExternalCode(CPlusPlusSource, Target(), DeviceAPI::None, code, name);
    }

    /** Return true if this container holds llvm bitcode linkable with
     * code generated for the target argument. The matching is done
     * on the architecture, bit width, and operating system
     * only. Features are ignored. If the container is for
     * Target::ArchUnkonwn, it applies to all architectures -- meaning
     * it is generic llvm bitcode. If the OS is OSUnknown, it applies
     * to all operationg systems. The bit width must match.
     *
     * Ignoring feature flags isn't too important since generally
     * ExternalCode will be constructed in a Generator which has
     * access to the feature flags in effect and can select code
     * appropriately. */
    bool is_for_cpu_target(const Target &host) const {
        return kind == LLVMBitcode &&
            (llvm_target.arch == Target::ArchUnknown || llvm_target.arch == host.arch) &&
            (llvm_target.os == Target::OSUnknown || llvm_target.os == host.os) &&
            (llvm_target.bits == host.bits);
    }

    /** True if this container holds code linkable with a code generated for a GPU. */
    bool is_for_device_api(DeviceAPI current_device) const {
        return kind == DeviceCode && device_code_kind == current_device;
    }

    /** True if this container holds C++ source code for inclusion in
     *  generating C++ output. */
    bool is_c_plus_plus_source() const { return kind == CPlusPlusSource; }

    /** Retrieve the bytes of external code held by this container. */
    const std::vector<uint8_t> &contents() const { return code; }

    /** Retrieve the name of this container. Used to ensure the same
     *  piece of external code is only included once in linkage. */
    const std::string &name() const { return nametag; }
};

}

#endif

#ifndef HALIDE_MODULUS_REMAINDER_H
#define HALIDE_MODULUS_REMAINDER_H

/** \file
 * Routines for statically determining what expressions are divisible by.
 */


namespace Halide {
namespace Internal {

/** The result of modulus_remainder analysis */
struct ModulusRemainder {
    ModulusRemainder() : modulus(0), remainder(0) {}
    ModulusRemainder(int m, int r) : modulus(m), remainder(r) {}
    int modulus, remainder;
};

/** For things like alignment analysis, often it's helpful to know
 * if an integer expression is some multiple of a constant plus
 * some other constant. For example, it is straight-forward to
 * deduce that ((10*x + 2)*(6*y - 3) - 1) is congruent to five
 * modulo six.
 *
 * We get the most information when the modulus is large. E.g. if
 * something is congruent to 208 modulo 384, then we also know it's
 * congruent to 0 mod 8, and we can possibly use it as an index for an
 * aligned load. If all else fails, we can just say that an integer is
 * congruent to zero modulo one.
 */
EXPORT ModulusRemainder modulus_remainder(Expr e);

/** If we have alignment information about external variables, we can
 * let the analysis know about that using this version of
 * modulus_remainder: */
EXPORT ModulusRemainder modulus_remainder(Expr e, const Scope<ModulusRemainder> &scope);

/** Reduce an expression modulo some integer. Returns true and assigns
 * to remainder if an answer could be found. */
///@{
EXPORT bool reduce_expr_modulo(Expr e, int modulus, int *remainder);
EXPORT bool reduce_expr_modulo(Expr e, int modulus, int *remainder, const Scope<ModulusRemainder> &scope);
///@}

EXPORT void modulus_remainder_test();

/** The greatest common divisor of two integers */
EXPORT int gcd(int, int);

/** The least common multiple of two integers */
EXPORT int lcm(int, int);

}
}

#endif
#ifndef HALIDE_OUTPUTS_H
#define HALIDE_OUTPUTS_H

/** \file
 *
 * Provides output functions to enable writing out various build
 * objects from Halide Module objects.
 */

#include <string>

namespace Halide {

/** A struct specifying a collection of outputs. Used as an argument
 * to Pipeline::compile_to and Func::compile_to and Module::compile. */
struct Outputs {
    /** The name of the emitted object file. Empty if no object file
     * output is desired. */
    std::string object_name;

    /** The name of the emitted text assembly file. Empty if no
     * assembly file output is desired. */
    std::string assembly_name;

    /** The name of the emitted llvm bitcode. Empty if no llvm bitcode
     * output is desired. */
    std::string bitcode_name;

    /** The name of the emitted llvm assembly. Empty if no llvm assembly
     * output is desired. */
    std::string llvm_assembly_name;

    /** The name of the emitted C header file. Empty if no C header file
     * output is desired. */
    std::string c_header_name;

    /** The name of the emitted C source file. Empty if no C source file
     * output is desired. */
    std::string c_source_name;

    /** The name of the emitted stmt file. Empty if no stmt file
     * output is desired. */
    std::string stmt_name;

    /** The name of the emitted stmt.html file. Empty if no stmt.html file
     * output is desired. */
    std::string stmt_html_name;

    /** The name of the emitted static library file. Empty if no static library
     * output is desired. */
    std::string static_library_name;

    /** Make a new Outputs struct that emits everything this one does
     * and also an object file with the given name. */
    Outputs object(const std::string &object_name) const {
        Outputs updated = *this;
        updated.object_name = object_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also an assembly file with the given name. */
    Outputs assembly(const std::string &assembly_name) const {
        Outputs updated = *this;
        updated.assembly_name = assembly_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also an llvm bitcode file with the given name. */
    Outputs bitcode(const std::string &bitcode_name) const {
        Outputs updated = *this;
        updated.bitcode_name = bitcode_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also an llvm assembly file with the given name. */
    Outputs llvm_assembly(const std::string &llvm_assembly_name) const {
        Outputs updated = *this;
        updated.llvm_assembly_name = llvm_assembly_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also a C header file with the given name. */
    Outputs c_header(const std::string &c_header_name) const {
        Outputs updated = *this;
        updated.c_header_name = c_header_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also a C source file with the given name. */
    Outputs c_source(const std::string &c_source_name) const {
        Outputs updated = *this;
        updated.c_source_name = c_source_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also a stmt file with the given name. */
    Outputs stmt(const std::string &stmt_name) const {
        Outputs updated = *this;
        updated.stmt_name = stmt_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also a stmt.html file with the given name. */
    Outputs stmt_html(const std::string &stmt_html_name) const {
        Outputs updated = *this;
        updated.stmt_html_name = stmt_html_name;
        return updated;
    }

    /** Make a new Outputs struct that emits everything this one does
     * and also a static library file with the given name. */
    Outputs static_library(const std::string &static_library_name) const {
        Outputs updated = *this;
        updated.static_library_name = static_library_name;
        return updated;
    }
};

}

#endif

namespace Halide {
namespace Internal {

/** Definition of an argument to a LoweredFunc. This is similar to
 * Argument, except it enables passing extra information useful to
 * some targets to LoweredFunc. */
struct LoweredArgument : public Argument {
    /** For scalar arguments, the modulus and remainder of this
     * argument. */
    ModulusRemainder alignment;

    LoweredArgument() {}
    LoweredArgument(const Argument &arg) : Argument(arg) {}
    LoweredArgument(const std::string &_name, Kind _kind, const Type &_type, uint8_t _dimensions,
                    Expr _def = Expr(),
                    Expr _min = Expr(),
                    Expr _max = Expr()) : Argument(_name, _kind, _type, _dimensions, _def, _min, _max) {}
};

/** Definition of a lowered function. This object provides a concrete
 * mapping between parameters used in the function body and their
 * declarations in the argument list. */
struct LoweredFunc {
    std::string name;

    /** Arguments referred to in the body of this function. */
    std::vector<LoweredArgument> args;

    /** Body of this function. */
    Stmt body;

    /** Type of linkage a function can have. */
    enum LinkageType {
        External, ///< Visible externally.
        ExternalPlusMetadata, ///< Visible externally. Argument metadata and an argv wrapper are also generated.
        Internal, ///< Not visible externally, similar to 'static' linkage in C.
    };

    /** The linkage of this function. */
    LinkageType linkage;

    /** The name-mangling choice for the function. Defaults to using
     * the Target. */
    NameMangling name_mangling;

    LoweredFunc(const std::string &name,
                const std::vector<LoweredArgument> &args,
                Stmt body,
                LinkageType linkage,
                NameMangling mangling = NameMangling::Default);
    LoweredFunc(const std::string &name,
                const std::vector<Argument> &args,
                Stmt body,
                LinkageType linkage,
                NameMangling mangling = NameMangling::Default);
};

}

namespace Internal {
struct ModuleContents;
}

/** A halide module. This represents IR containing lowered function
 * definitions and buffers. */
class Module {
    Internal::IntrusivePtr<Internal::ModuleContents> contents;

public:
    EXPORT Module(const std::string &name, const Target &target);

    /** Get the target this module has been lowered for. */
    EXPORT const Target &target() const;

    /** The name of this module. This is used as the default filename
     * for output operations. */
    EXPORT const std::string &name() const;

    /** The declarations contained in this module. */
    // @{
    EXPORT const std::vector<Buffer<>> &buffers() const;
    EXPORT const std::vector<Internal::LoweredFunc> &functions() const;
    EXPORT std::vector<Internal::LoweredFunc> &functions();
    EXPORT const std::vector<Module> &submodules() const;
    EXPORT const std::vector<ExternalCode> &external_code() const;
    // @}

    /** Return the function with the given name. If no such function
    * exists in this module, assert. */
    EXPORT Internal::LoweredFunc get_function_by_name(const std::string &name) const;

    /** Add a declaration to this module. */
    // @{
    EXPORT void append(const Buffer<> &buffer);
    EXPORT void append(const Internal::LoweredFunc &function);
    EXPORT void append(const Module &module);
    EXPORT void append(const ExternalCode &external_code);
    // @}

    /** Compile a halide Module to variety of outputs, depending on
     * the fields set in output_files. */
    EXPORT void compile(const Outputs &output_files) const;

    /** Compile a halide Module to in-memory object code. Currently
     * only supports LLVM based compilation, but should be extended to
     * handle source code backends. */
    EXPORT Buffer<uint8_t> compile_to_buffer() const;

    /** Return a new module with all submodules compiled to buffers on
     * on the result Module. */
    EXPORT Module resolve_submodules() const;
};

/** Link a set of modules together into one module. */
EXPORT Module link_modules(const std::string &name, const std::vector<Module> &modules);

/** Create an object file containing the Halide runtime for a given
 * target. For use with Target::NoRuntime. */
EXPORT void compile_standalone_runtime(const std::string &object_filename, Target t);

/** Create an object and/or static library file containing the Halide runtime for a given
 * target. For use with Target::NoRuntime. Return an Outputs with just the actual
 * outputs filled in (typically, object_name and/or static_library_name).
 */
EXPORT Outputs compile_standalone_runtime(const Outputs &output_files, Target t);

typedef std::function<Module(const std::string &, const Target &)> ModuleProducer;

EXPORT void compile_multitarget(const std::string &fn_name,
                                const Outputs &output_files,
                                const std::vector<Target> &targets,
                                ModuleProducer module_producer,
                                const std::map<std::string, std::string> &suffixes = {});

}

#endif
#ifndef HALIDE_PIPELINE_H
#define HALIDE_PIPELINE_H

/** \file
 *
 * Defines the front-end class representing an entire Halide imaging
 * pipeline.
 */

#include <vector>


namespace Halide {

struct Argument;
class Func;
struct Outputs;
struct PipelineContents;

namespace Internal {
class IRMutator;
}  // namespace Internal

/**
 * Used to determine if the output printed to file should be as a normal string
 * or as an HTML file which can be opened in a browerser and manipulated via JS and CSS.*/
enum StmtOutputFormat {
     Text,
     HTML
};

namespace {
// Helper for deleting custom lowering passes. In the header so that
// it goes in user code on windows, where you can have multiple heaps.
template<typename T>
void delete_lowering_pass(T *pass) {
    delete pass;
}
}  // namespace

/** A custom lowering pass. See Pipeline::add_custom_lowering_pass. */
struct CustomLoweringPass {
    Internal::IRMutator *pass;
    void (*deleter)(Internal::IRMutator *);
};

struct JITExtern;

/** A class representing a Halide pipeline. Constructed from the Func
 * or Funcs that it outputs. */
class Pipeline {
    Internal::IntrusivePtr<PipelineContents> contents;

    std::vector<Argument> infer_arguments(Internal::Stmt body);
    std::vector<const void *> prepare_jit_call_arguments(Realization dst, const Target &target);

    static std::vector<Internal::JITModule> make_externs_jit_module(const Target &target,
                                                                    std::map<std::string, JITExtern> &externs_in_out);

public:
    /** Make an undefined Pipeline object. */
    EXPORT Pipeline();

    /** Make a pipeline that computes the given Func. Schedules the
     * Func compute_root(). */
    EXPORT Pipeline(Func output);

    /** Make a pipeline that computes the givens Funcs as
     * outputs. Schedules the Funcs compute_root(). */
    EXPORT Pipeline(const std::vector<Func> &outputs);

    /** Get the Funcs this pipeline outputs. */
    EXPORT std::vector<Func> outputs() const;

    /** Generate a schedule for the pipeline. */
    //@{
    EXPORT std::string auto_schedule(const Target &target,
                                     const MachineParams &arch_params);
    EXPORT std::string auto_schedule(const Target &target);
    //@}

    /** Return handle to the index-th Func within the pipeline based on the
     * realization order. */
    EXPORT Func get_func(size_t index);

    /** Compile and generate multiple target files with single call.
     * Deduces target files based on filenames specified in
     * output_files struct.
     */
    EXPORT void compile_to(const Outputs &output_files,
                           const std::vector<Argument> &args,
                           const std::string &fn_name,
                           const Target &target);

    /** Statically compile a pipeline to llvm bitcode, with the given
     * filename (which should probably end in .bc), type signature,
     * and C function name. If you're compiling a pipeline with a
     * single output Func, see also Func::compile_to_bitcode. */
    EXPORT void compile_to_bitcode(const std::string &filename,
                                   const std::vector<Argument> &args,
                                   const std::string &fn_name,
                                   const Target &target = get_target_from_environment());

    /** Statically compile a pipeline to llvm assembly, with the given
     * filename (which should probably end in .ll), type signature,
     * and C function name. If you're compiling a pipeline with a
     * single output Func, see also Func::compile_to_llvm_assembly. */
    EXPORT void compile_to_llvm_assembly(const std::string &filename,
                                         const std::vector<Argument> &args,
                                         const std::string &fn_name,
                                         const Target &target = get_target_from_environment());

    /** Statically compile a pipeline with multiple output functions to an
     * object file, with the given filename (which should probably end in
     * .o or .obj), type signature, and C function name (which defaults to
     * the same name as this halide function. You probably don't want to
     * use this directly; call compile_to_static_library or compile_to_file instead. */
    EXPORT void compile_to_object(const std::string &filename,
                                  const std::vector<Argument> &,
                                  const std::string &fn_name,
                                  const Target &target = get_target_from_environment());

    /** Emit a header file with the given filename for a pipeline. The
     * header will define a function with the type signature given by
     * the second argument, and a name given by the third. You don't
     * actually have to have defined any of these functions yet to
     * call this. You probably don't want to use this directly; call
     * compile_to_static_library or compile_to_file instead. */
    EXPORT void compile_to_header(const std::string &filename,
                                  const std::vector<Argument> &,
                                  const std::string &fn_name,
                                  const Target &target = get_target_from_environment());

    /** Statically compile a pipeline to text assembly equivalent to
     * the object file generated by compile_to_object. This is useful
     * for checking what Halide is producing without having to
     * disassemble anything, or if you need to feed the assembly into
     * some custom toolchain to produce an object file. */
    EXPORT void compile_to_assembly(const std::string &filename,
                                    const std::vector<Argument> &args,
                                    const std::string &fn_name,
                                    const Target &target = get_target_from_environment());

    /** Statically compile a pipeline to C source code. This is useful
     * for providing fallback code paths that will compile on many
     * platforms. Vectorization will fail, and parallelization will
     * produce serial code. */
    EXPORT void compile_to_c(const std::string &filename,
                             const std::vector<Argument> &,
                             const std::string &fn_name,
                             const Target &target = get_target_from_environment());

    /** Statically compile a pipeline to Tiramisu source code. */
    EXPORT void compile_to_tiramisu(const std::string &filename,
                                    const std::string &fn_name,
                                    const Target &target = get_target_from_environment());

    /** Write out an internal representation of lowered code. Useful
     * for analyzing and debugging scheduling. Can emit html or plain
     * text. */
    EXPORT void compile_to_lowered_stmt(const std::string &filename,
                                        const std::vector<Argument> &args,
                                        StmtOutputFormat fmt = Text,
                                        const Target &target = get_target_from_environment());

    /** Write out the loop nests specified by the schedule for this
     * Pipeline's Funcs. Helpful for understanding what a schedule is
     * doing. */
    EXPORT void print_loop_nest();

    /** Compile to object file and header pair, with the given
     * arguments. */
    EXPORT void compile_to_file(const std::string &filename_prefix,
                                const std::vector<Argument> &args,
                                const std::string &fn_name,
                                const Target &target = get_target_from_environment());

    /** Compile to static-library file and header pair, with the given
     * arguments. */
    EXPORT void compile_to_static_library(const std::string &filename_prefix,
                                          const std::vector<Argument> &args,
                                          const std::string &fn_name,
                                          const Target &target = get_target_from_environment());

    /** Compile to static-library file and header pair once for each target;
     * each resulting function will be considered (in order) via halide_can_use_target_features()
     * at runtime, with the first appropriate match being selected for subsequent use.
     * This is typically useful for specializations that may vary unpredictably by machine
     * (e.g., SSE4.1/AVX/AVX2 on x86 desktop machines).
     * All targets must have identical arch-os-bits.
     */
    EXPORT void compile_to_multitarget_static_library(const std::string &filename_prefix,
                                                      const std::vector<Argument> &args,
                                                      const std::vector<Target> &targets);

    /** Create an internal representation of lowered code as a self
     * contained Module suitable for further compilation. */
    EXPORT Module compile_to_module(const std::vector<Argument> &args,
                                    const std::string &fn_name,
                                    const Target &target = get_target_from_environment(),
                                    const Internal::LoweredFunc::LinkageType linkage_type = Internal::LoweredFunc::ExternalPlusMetadata);

   /** Eagerly jit compile the function to machine code. This
     * normally happens on the first call to realize. If you're
     * running your halide pipeline inside time-sensitive code and
     * wish to avoid including the time taken to compile a pipeline,
     * then you can call this ahead of time. Returns the raw function
     * pointer to the compiled pipeline. Default is to use the Target
     * returned from Halide::get_jit_target_from_environment()
     */
     EXPORT void *compile_jit(const Target &target = get_jit_target_from_environment());

    /** Set the error handler function that be called in the case of
     * runtime errors during halide pipelines. If you are compiling
     * statically, you can also just define your own function with
     * signature
     \code
     extern "C" void halide_error(void *user_context, const char *);
     \endcode
     * This will clobber Halide's version.
     */
    EXPORT void set_error_handler(void (*handler)(void *, const char *));

    /** Set a custom malloc and free for halide to use. Malloc should
     * return 32-byte aligned chunks of memory, and it should be safe
     * for Halide to read slightly out of bounds (up to 8 bytes before
     * the start or beyond the end). If compiling statically, routines
     * with appropriate signatures can be provided directly
    \code
     extern "C" void *halide_malloc(void *, size_t)
     extern "C" void halide_free(void *, void *)
     \endcode
     * These will clobber Halide's versions. See \file HalideRuntime.h
     * for declarations.
     */
    EXPORT void set_custom_allocator(void *(*malloc)(void *, size_t),
                                     void (*free)(void *, void *));

    /** Set a custom task handler to be called by the parallel for
     * loop. It is useful to set this if you want to do some
     * additional bookkeeping at the granularity of parallel
     * tasks. The default implementation does this:
     \code
     extern "C" int halide_do_task(void *user_context,
                                   int (*f)(void *, int, uint8_t *),
                                   int idx, uint8_t *state) {
         return f(user_context, idx, state);
     }
     \endcode
     * If you are statically compiling, you can also just define your
     * own version of the above function, and it will clobber Halide's
     * version.
     *
     * If you're trying to use a custom parallel runtime, you probably
     * don't want to call this. See instead \ref Func::set_custom_do_par_for .
    */
    EXPORT void set_custom_do_task(
        int (*custom_do_task)(void *, int (*)(void *, int, uint8_t *),
                              int, uint8_t *));

    /** Set a custom parallel for loop launcher. Useful if your app
     * already manages a thread pool. The default implementation is
     * equivalent to this:
     \code
     extern "C" int halide_do_par_for(void *user_context,
                                      int (*f)(void *, int, uint8_t *),
                                      int min, int extent, uint8_t *state) {
         int exit_status = 0;
         parallel for (int idx = min; idx < min+extent; idx++) {
             int job_status = halide_do_task(user_context, f, idx, state);
             if (job_status) exit_status = job_status;
         }
         return exit_status;
     }
     \endcode
     *
     * However, notwithstanding the above example code, if one task
     * fails, we may skip over other tasks, and if two tasks return
     * different error codes, we may select one arbitrarily to return.
     *
     * If you are statically compiling, you can also just define your
     * own version of the above function, and it will clobber Halide's
     * version.
     */
    EXPORT void set_custom_do_par_for(
        int (*custom_do_par_for)(void *, int (*)(void *, int, uint8_t *), int,
                                 int, uint8_t *));

    /** Set custom routines to call when tracing is enabled. Call this
     * on the output Func of your pipeline. This then sets custom
     * routines for the entire pipeline, not just calls to this
     * Func.
     *
     * If you are statically compiling, you can also just define your
     * own versions of the tracing functions (see HalideRuntime.h),
     * and they will clobber Halide's versions. */
    EXPORT void set_custom_trace(int (*trace_fn)(void *, const halide_trace_event_t *));

    /** Set the function called to print messages from the runtime.
     * If you are compiling statically, you can also just define your
     * own function with signature
     \code
     extern "C" void halide_print(void *user_context, const char *);
     \endcode
     * This will clobber Halide's version.
     */
    EXPORT void set_custom_print(void (*handler)(void *, const char *));

    /** Install a set of external C functions or Funcs to satisfy
     * dependencies introduced by HalideExtern and define_extern
     * mechanisms. These will be used by calls to realize,
     * infer_bounds, and compile_jit. */
    EXPORT void set_jit_externs(const std::map<std::string, JITExtern> &externs);

    /** Return the map of previously installed externs. Is an empty
     * map unless set otherwise. */
    EXPORT const std::map<std::string, JITExtern> &get_jit_externs();

    /** Get a struct containing the currently set custom functions
     * used by JIT. */
    EXPORT const Internal::JITHandlers &jit_handlers();

    /** Add a custom pass to be used during lowering. It is run after
     * all other lowering passes. Can be used to verify properties of
     * the lowered Stmt, instrument it with extra code, or otherwise
     * modify it. The Func takes ownership of the pass, and will call
     * delete on it when the Func goes out of scope. So don't pass a
     * stack object, or share pass instances between multiple
     * Funcs. */
    template<typename T>
    void add_custom_lowering_pass(T *pass) {
        // Template instantiate a custom deleter for this type, then
        // cast it to a deleter that takes a IRMutator *. The custom
        // deleter lives in user code, so that deletion is on the same
        // heap as construction (I hate Windows).
        void (*deleter)(Internal::IRMutator *) =
            (void (*)(Internal::IRMutator *))(&delete_lowering_pass<T>);
        add_custom_lowering_pass(pass, deleter);
    }

    /** Add a custom pass to be used during lowering, with the
     * function that will be called to delete it also passed in. Set
     * it to nullptr if you wish to retain ownership of the object. */
    EXPORT void add_custom_lowering_pass(Internal::IRMutator *pass,
                                         void (*deleter)(Internal::IRMutator *));

    /** Remove all previously-set custom lowering passes */
    EXPORT void clear_custom_lowering_passes();

    /** Get the custom lowering passes. */
    EXPORT const std::vector<CustomLoweringPass> &custom_lowering_passes();

    /** See Func::realize */
    // @{
    EXPORT Realization realize(std::vector<int32_t> sizes, const Target &target = Target());
    EXPORT Realization realize(int x_size, int y_size, int z_size, int w_size,
                               const Target &target = Target());
    EXPORT Realization realize(int x_size, int y_size, int z_size,
                               const Target &target = Target());
    EXPORT Realization realize(int x_size, int y_size,
                               const Target &target = Target());
    EXPORT Realization realize(int x_size,
                               const Target &target = Target());
    EXPORT Realization realize(const Target &target = Target());
    // @}

    /** Evaluate this Pipeline into an existing allocated buffer or
     * buffers. If the buffer is also one of the arguments to the
     * function, strange things may happen, as the pipeline isn't
     * necessarily safe to run in-place. The realization should
     * contain one Buffer per tuple component per output Func. For
     * each individual output Func, all Buffers must have the same
     * shape, but the shape can vary across the different output
     * Funcs. This form of realize does *not* automatically copy data
     * back from the GPU. */
    EXPORT void realize(Realization dst, const Target &target = Target());

    /** For a given size of output, or a given set of output buffers,
     * determine the bounds required of all unbound ImageParams
     * referenced. Communicates the result by allocating new buffers
     * of the appropriate size and binding them to the unbound
     * ImageParams. */
    // @{
    EXPORT void infer_input_bounds(int x_size = 0, int y_size = 0, int z_size = 0, int w_size = 0);
    EXPORT void infer_input_bounds(Realization dst);
    // @}

    /** Infer the arguments to the Pipeline, sorted into a canonical order:
     * all buffers (sorted alphabetically by name), followed by all non-buffers
     * (sorted alphabetically by name).
     This lets you write things like:
     \code
     pipeline.compile_to_assembly("/dev/stdout", pipeline.infer_arguments());
     \endcode
     */
    EXPORT std::vector<Argument> infer_arguments();

    /** Check if this pipeline object is defined. That is, does it
     * have any outputs? */
    EXPORT bool defined() const;

    /** Invalidate any internal cached state, e.g. because Funcs have
     * been rescheduled. */
    EXPORT void invalidate_cache();

private:
    std::string generate_function_name() const;
};

struct ExternSignature {
private:
    Type ret_type_;       // Only meaningful if is_void_return is false; must be default value otherwise
    bool is_void_return_{false};
    std::vector<Type> arg_types_;

public:
    ExternSignature() = default;

    ExternSignature(const Type &ret_type, bool is_void_return, const std::vector<Type> &arg_types)
        : ret_type_(ret_type),
          is_void_return_(is_void_return),
          arg_types_(arg_types) {
        internal_assert(!(is_void_return && ret_type != Type()));
    }

    template <typename RT, typename... Args>
    ExternSignature(RT (*f)(Args... args))
        : ret_type_(type_of<RT>()),
          is_void_return_(std::is_void<RT>::value),
          arg_types_({type_of<Args>()...}) {
    }

    const Type &ret_type() const {
        internal_assert(!is_void_return());
        return ret_type_;
    }

    bool is_void_return() const {
        return is_void_return_;
    }

    const std::vector<Type> &arg_types() const {
        return arg_types_;
    }
};

struct ExternCFunction {
private:
    void *address_{nullptr};
    ExternSignature signature_;

public:
    ExternCFunction() = default;

    ExternCFunction(void *address, const ExternSignature &signature)
        : address_(address), signature_(signature) {}

    template <typename RT, typename... Args>
    ExternCFunction(RT (*f)(Args... args)) : ExternCFunction((void *)f, ExternSignature(f)) {}

    void *address() const { return address_; }
    const ExternSignature &signature() const { return signature_; }
};

struct JITExtern {
private:
    // Note that exactly one of pipeline_ and extern_c_function_
    // can be set in a given JITExtern instance.
    Pipeline pipeline_;
    ExternCFunction extern_c_function_;

public:
    EXPORT JITExtern(Pipeline pipeline);
    EXPORT JITExtern(Func func);
    EXPORT JITExtern(const ExternCFunction &extern_c_function);

    template <typename RT, typename... Args>
    JITExtern(RT (*f)(Args... args)) : JITExtern(ExternCFunction(f)) {}

    const Pipeline &pipeline() const { return pipeline_; }
    const ExternCFunction &extern_c_function() const { return extern_c_function_; }
};

}  // namespace Halide

#endif

#include <map>

namespace Halide {

/** A class that can represent Vars or RVars. Used for reorder calls
 * which can accept a mix of either. */
struct VarOrRVar {
    VarOrRVar(const std::string &n, bool r) : var(n), rvar(n), is_rvar(r) {}
    VarOrRVar(const Var &v) : var(v), is_rvar(false) {}
    VarOrRVar(const RVar &r) : rvar(r), is_rvar(true) {}
    VarOrRVar(const RDom &r) : rvar(RVar(r)), is_rvar(true) {}

    const std::string &name() const {
        if (is_rvar) return rvar.name();
        else return var.name();
    }

    Var var;
    RVar rvar;
    bool is_rvar;
};

class ImageParam;

namespace Internal {
struct Split;
struct StorageDim;
}

/** A single definition of a Func. May be a pure or update definition. */
class Stage {
    Internal::Definition definition;
    std::string stage_name;
    /** Pure Vars of the Function (from the init definition). */
    std::vector<Var> dim_vars;
    /** This is just a reference to the FuncSchedule owned by the Function
     * associated with this Stage. */
    Internal::FuncSchedule func_schedule;

    void set_dim_type(VarOrRVar var, Internal::ForType t);
    void set_dim_device_api(VarOrRVar var, DeviceAPI device_api);
    void split(const std::string &old, const std::string &outer, const std::string &inner,
               Expr factor, bool exact, TailStrategy tail);
    void remove(const std::string &var);
    Stage &purify(VarOrRVar old_name, VarOrRVar new_name);

public:
    Stage(Internal::Definition d, const std::string &n, const std::vector<Var> &args,
          const Internal::FuncSchedule &func_s)
            : definition(d), stage_name(n), dim_vars(args), func_schedule(func_s) {
        internal_assert(definition.args().size() == dim_vars.size());
        definition.schedule().touched() = true;
    }

    Stage(Internal::Definition d, const std::string &n, const std::vector<std::string> &args,
          const Internal::FuncSchedule &func_s)
            : definition(d), stage_name(n), func_schedule(func_s) {
        definition.schedule().touched() = true;

        std::vector<Var> dim_vars(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            dim_vars[i] = Var(args[i]);
        }
        internal_assert(definition.args().size() == dim_vars.size());
    }

    /** Return the current StageSchedule associated with this Stage. For
     * introspection only: to modify schedule, use the Func interface. */
    const Internal::StageSchedule &get_schedule() const { return definition.schedule(); }

    /** Return a string describing the current var list taking into
     * account all the splits, reorders, and tiles. */
    EXPORT std::string dump_argument_list() const;

    /** Return the name of this stage, e.g. "f.update(2)" */
    EXPORT const std::string &name() const;

    /** Calling rfactor() on an associative update definition a Func will split
     * the update into an intermediate which computes the partial results and
     * replaces the current update definition with a new definition which merges
     * the partial results. If called on a init/pure definition, this will
     * throw an error. rfactor() will automatically infer the associative reduction
     * operator and identity of the operator. If it can't prove the operation
     * is associative or if it cannot find an identity for that operator, this
     * will throw an error. In addition, commutativity of the operator is required
     * if rfactor() is called on the inner dimension but excluding the outer
     * dimensions.
     *
     * rfactor() takes as input 'preserved', which is a list of <RVar, Var> pairs.
     * The rvars not listed in 'preserved' are removed from the original Func and
     * are lifted to the intermediate Func. The remaining rvars (the ones in
     * 'preserved') are made pure in the intermediate Func. The intermediate Func's
     * update definition inherits all scheduling directives (e.g. split,fuse, etc.)
     * applied to the original Func's update definition. The loop order of the
     * intermediate Func's update definition is the same as the original, although
     * the RVars in 'preserved' are replaced by the new pure Vars. The loop order of the
     * intermediate Func's init definition from innermost to outermost is the args'
     * order of the original Func's init definition followed by the new pure Vars.
     *
     * The intermediate Func also inherits storage order from the original Func
     * with the new pure Vars added to the outermost.
     *
     * For example, f.update(0).rfactor({{r.y, u}}) would rewrite a pipeline like this:
     \code
     f(x, y) = 0;
     f(x, y) += g(r.x, r.y);
     \endcode
     * into a pipeline like this:
     \code
     f_intm(x, y, u) = 0;
     f_intm(x, y, u) += g(r.x, u);

     f(x, y) = 0;
     f(x, y) += f_intm(x, y, r.y);
     \endcode
     *
     * This has a variety of uses. You can use it to split computation of an associative reduction:
     \code
     f(x, y) = 10;
     RDom r(0, 96);
     f(x, y) = max(f(x, y), g(x, y, r.x));
     f.update(0).split(r.x, rxo, rxi, 8).reorder(y, x).parallel(x);
     f.update(0).rfactor({{rxo, u}}).compute_root().parallel(u).update(0).parallel(u);
     \endcode
     *
     *, which is equivalent to:
     \code
     parallel for u = 0 to 11:
       for y:
         for x:
           f_intm(x, y, u) = -inf
     parallel for x:
       for y:
         parallel for u = 0 to 11:
           for rxi = 0 to 7:
             f_intm(x, y, u) = max(f_intm(x, y, u), g(8*u + rxi))
     for y:
       for x:
         f(x, y) = 10
     parallel for x:
       for y:
         for rxo = 0 to 11:
           f(x, y) = max(f(x, y), f_intm(x, y, u))
     \endcode
     *
     */
    // @{
    EXPORT Func rfactor(std::vector<std::pair<RVar, Var>> preserved);
    EXPORT Func rfactor(RVar r, Var v);
    // @}

    /** Scheduling calls that control how the domain of this stage is
     * traversed. See the documentation for Func for the meanings. */
    // @{

    EXPORT Stage &split(VarOrRVar old, VarOrRVar outer, VarOrRVar inner, Expr factor, TailStrategy tail = TailStrategy::Auto);
    EXPORT Stage &fuse(VarOrRVar inner, VarOrRVar outer, VarOrRVar fused);
    EXPORT Stage &serial(VarOrRVar var);
    EXPORT Stage &parallel(VarOrRVar var);
    EXPORT Stage &vectorize(VarOrRVar var);
    EXPORT Stage &unroll(VarOrRVar var);
    EXPORT Stage &parallel(VarOrRVar var, Expr task_size, TailStrategy tail = TailStrategy::Auto);
    EXPORT Stage &vectorize(VarOrRVar var, Expr factor, TailStrategy tail = TailStrategy::Auto);
    EXPORT Stage &unroll(VarOrRVar var, Expr factor, TailStrategy tail = TailStrategy::Auto);
    EXPORT Stage &tile(VarOrRVar x, VarOrRVar y,
                       VarOrRVar xo, VarOrRVar yo,
                       VarOrRVar xi, VarOrRVar yi, Expr
                       xfactor, Expr yfactor,
                       TailStrategy tail = TailStrategy::Auto);
    EXPORT Stage &tile(VarOrRVar x, VarOrRVar y,
                       VarOrRVar xi, VarOrRVar yi,
                       Expr xfactor, Expr yfactor,
                       TailStrategy tail = TailStrategy::Auto);
    EXPORT Stage &reorder(const std::vector<VarOrRVar> &vars);

    template <typename... Args>
    NO_INLINE typename std::enable_if<Internal::all_are_convertible<VarOrRVar, Args...>::value, Stage &>::type
    reorder(VarOrRVar x, VarOrRVar y, Args&&... args) {
        std::vector<VarOrRVar> collected_args{x, y, std::forward<Args>(args)...};
        return reorder(collected_args);
    }

    EXPORT Stage &rename(VarOrRVar old_name, VarOrRVar new_name);
    EXPORT Stage specialize(Expr condition);
    EXPORT void specialize_fail(const std::string &message);

    EXPORT Stage &gpu_threads(VarOrRVar thread_x, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_threads(VarOrRVar thread_x, VarOrRVar thread_y, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_threads(VarOrRVar thread_x, VarOrRVar thread_y, VarOrRVar thread_z, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_single_thread(DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Stage &gpu_blocks(VarOrRVar block_x, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_blocks(VarOrRVar block_x, VarOrRVar block_y, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_blocks(VarOrRVar block_x, VarOrRVar block_y, VarOrRVar block_z, DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Stage &gpu(VarOrRVar block_x, VarOrRVar thread_x, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu(VarOrRVar block_x, VarOrRVar block_y,
                      VarOrRVar thread_x, VarOrRVar thread_y,
                      DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu(VarOrRVar block_x, VarOrRVar block_y, VarOrRVar block_z,
                      VarOrRVar thread_x, VarOrRVar thread_y, VarOrRVar thread_z,
                      DeviceAPI device_api = DeviceAPI::Default_GPU);

    // TODO(psuriana): For now we need to expand "tx" into Var and RVar versions
    // due to conflict with the deprecated interfaces since Var can be implicitly
    // converted into either VarOrRVar or Expr. Merge this later once we remove
    // the deprecated interfaces.
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar bx, Var tx, Expr x_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar bx, RVar tx, Expr x_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar tx, Expr x_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y,
                           VarOrRVar bx, VarOrRVar by,
                           VarOrRVar tx, VarOrRVar ty,
                           Expr x_size, Expr y_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y,
                           VarOrRVar tx, Var ty,
                           Expr x_size, Expr y_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y,
                           VarOrRVar tx, RVar ty,
                           Expr x_size, Expr y_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y, VarOrRVar z,
                           VarOrRVar bx, VarOrRVar by, VarOrRVar bz,
                           VarOrRVar tx, VarOrRVar ty, VarOrRVar tz,
                           Expr x_size, Expr y_size, Expr z_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y, VarOrRVar z,
                           VarOrRVar tx, VarOrRVar ty, VarOrRVar tz,
                           Expr x_size, Expr y_size, Expr z_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);

    // If we mark these as deprecated, some build environments will complain
    // about the internal-only calls. Since these are rarely used outside
    // Func itself, we'll just comment them as deprecated for now.
    // HALIDE_ATTRIBUTE_DEPRECATED("This form of gpu_tile() is deprecated.")
    EXPORT Stage &gpu_tile(VarOrRVar x, Expr x_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);
    // HALIDE_ATTRIBUTE_DEPRECATED("This form of gpu_tile() is deprecated.")
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y,
                           Expr x_size, Expr y_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);
    // HALIDE_ATTRIBUTE_DEPRECATED("This form of gpu_tile() is deprecated.")
    EXPORT Stage &gpu_tile(VarOrRVar x, VarOrRVar y, VarOrRVar z,
                           Expr x_size, Expr y_size, Expr z_size,
                           TailStrategy tail = TailStrategy::Auto,
                           DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Stage &allow_race_conditions();

    EXPORT Stage &hexagon(VarOrRVar x = Var::outermost());
    EXPORT Stage &prefetch(const Func &f, VarOrRVar var, Expr offset = 1,
                           PrefetchBoundStrategy strategy = PrefetchBoundStrategy::GuardWithIf);
    EXPORT Stage &prefetch(const Internal::Parameter &param, VarOrRVar var, Expr offset = 1,
                           PrefetchBoundStrategy strategy = PrefetchBoundStrategy::GuardWithIf);
    template<typename T>
    Stage &prefetch(const T &image, VarOrRVar var, Expr offset = 1,
                    PrefetchBoundStrategy strategy = PrefetchBoundStrategy::GuardWithIf) {
        return prefetch(image.parameter(), var, offset, strategy);
    }
    // @}
};

// For backwards compatibility, keep the ScheduleHandle name.
typedef Stage ScheduleHandle;


class FuncTupleElementRef;

/** A fragment of front-end syntax of the form f(x, y, z), where x, y,
 * z are Vars or Exprs. If could be the left hand side of a definition or
 * an update definition, or it could be a call to a function. We don't know
 * until we see how this object gets used.
 */
class FuncRef {
    Internal::Function func;
    int implicit_placeholder_pos;
    int implicit_count;
    std::vector<Expr> args;
    std::vector<Expr> args_with_implicit_vars(const std::vector<Expr> &e) const;

    /** Helper for function update by Tuple. If the function does not
     * already have a pure definition, init_val will be used as RHS of
     * each tuple element in the initial function definition. */
    template <typename BinaryOp>
    Stage func_ref_update(const Tuple &e, int init_val);

    /** Helper for function update by Expr. If the function does not
     * already have a pure definition, init_val will be used as RHS in
     * the initial function definition. */
    template <typename BinaryOp>
    Stage func_ref_update(Expr e, int init_val);

public:
    FuncRef(Internal::Function, const std::vector<Expr> &,
                int placeholder_pos = -1, int count = 0);
    FuncRef(Internal::Function, const std::vector<Var> &,
                int placeholder_pos = -1, int count = 0);

    /** Use this as the left-hand-side of a definition or an update definition
     * (see \ref RDom).
     */
    EXPORT Stage operator=(Expr);

    /** Use this as the left-hand-side of a definition or an update definition
     * for a Func with multiple outputs. */
    EXPORT Stage operator=(const Tuple &);

    /** Define a stage that adds the given expression to this Func. If the
     * expression refers to some RDom, this performs a sum reduction of the
     * expression over the domain. If the function does not already have a
     * pure definition, this sets it to zero.
     */
    // @{
    EXPORT Stage operator+=(Expr);
    EXPORT Stage operator+=(const Tuple &);
    EXPORT Stage operator+=(const FuncRef &);
    // @}

    /** Define a stage that adds the negative of the given expression to this
     * Func. If the expression refers to some RDom, this performs a sum reduction
     * of the negative of the expression over the domain. If the function does
     * not already have a pure definition, this sets it to zero.
     */
    // @{
    EXPORT Stage operator-=(Expr);
    EXPORT Stage operator-=(const Tuple &);
    EXPORT Stage operator-=(const FuncRef &);
    // @}

    /** Define a stage that multiplies this Func by the given expression. If the
     * expression refers to some RDom, this performs a product reduction of the
     * expression over the domain. If the function does not already have a pure
     * definition, this sets it to 1.
     */
    // @{
    EXPORT Stage operator*=(Expr);
    EXPORT Stage operator*=(const Tuple &);
    EXPORT Stage operator*=(const FuncRef &);
    // @}

    /** Define a stage that divides this Func by the given expression.
     * If the expression refers to some RDom, this performs a product
     * reduction of the inverse of the expression over the domain. If the
     * function does not already have a pure definition, this sets it to 1.
     */
    // @{
    EXPORT Stage operator/=(Expr);
    EXPORT Stage operator/=(const Tuple &);
    EXPORT Stage operator/=(const FuncRef &);
    // @}

    /* Override the usual assignment operator, so that
     * f(x, y) = g(x, y) defines f.
     */
    EXPORT Stage operator=(const FuncRef &);

    /** Use this as a call to the function, and not the left-hand-side
     * of a definition. Only works for single-output Funcs. */
    EXPORT operator Expr() const;

    /** When a FuncRef refers to a function that provides multiple
     * outputs, you can access each output as an Expr using
     * operator[].
     */
    EXPORT FuncTupleElementRef operator[](int) const;

    /** How many outputs does the function this refers to produce. */
    EXPORT size_t size() const;

    /** What function is this calling? */
    EXPORT Internal::Function function() const {return func;}
};

/** Explicit overloads of min and max for FuncRef. These exist to
 * disambiguate calls to min on FuncRefs when a user has pulled both
 * Halide::min and std::min into their namespace. */
// @{
inline Expr min(FuncRef a, FuncRef b) {return min(Expr(std::move(a)), Expr(std::move(b)));}
inline Expr max(FuncRef a, FuncRef b) {return max(Expr(std::move(a)), Expr(std::move(b)));}
// @}

/** A fragment of front-end syntax of the form f(x, y, z)[index], where x, y,
 * z are Vars or Exprs. If could be the left hand side of an update
 * definition, or it could be a call to a function. We don't know
 * until we see how this object gets used.
 */
class FuncTupleElementRef {
    FuncRef func_ref;
    std::vector<Expr> args; // args to the function
    int idx;                // Index to function outputs

    /** Helper function that generates a Tuple where element at 'idx' is set
     * to 'e' and the rests are undef. */
    Tuple values_with_undefs(Expr e) const;

public:
    FuncTupleElementRef(const FuncRef &ref, const std::vector<Expr>& args, int idx);

    /** Use this as the left-hand-side of an update definition of Tuple
     * component 'idx' of a Func (see \ref RDom). The function must
     * already have an initial definition.
     */
    EXPORT Stage operator=(Expr e);


    /** Define a stage that adds the given expression to Tuple component 'idx'
     * of this Func. The other Tuple components are unchanged. If the expression
     * refers to some RDom, this performs a sum reduction of the expression over
     * the domain. The function must already have an initial definition.
     */
    EXPORT Stage operator+=(Expr e);

    /** Define a stage that adds the negative of the given expression to Tuple
     * component 'idx' of this Func. The other Tuple components are unchanged.
     * If the expression refers to some RDom, this performs a sum reduction of
     * the negative of the expression over the domain. The function must already
     * have an initial definition.
     */
    EXPORT Stage operator-=(Expr e);

    /** Define a stage that multiplies Tuple component 'idx' of this Func by
     * the given expression. The other Tuple components are unchanged. If the
     * expression refers to some RDom, this performs a product reduction of
     * the expression over the domain. The function must already have an
     * initial definition.
     */
    EXPORT Stage operator*=(Expr e);

    /** Define a stage that divides Tuple component 'idx' of this Func by
     * the given expression. The other Tuple components are unchanged.
     * If the expression refers to some RDom, this performs a product
     * reduction of the inverse of the expression over the domain. The function
     * must already have an initial definition.
     */
    EXPORT Stage operator/=(Expr e);

    /* Override the usual assignment operator, so that
     * f(x, y)[index] = g(x, y) defines f.
     */
    EXPORT Stage operator=(const FuncRef &e);

    /** Use this as a call to Tuple component 'idx' of a Func, and not the
     * left-hand-side of a definition. */
    EXPORT operator Expr() const;

    /** What function is this calling? */
    EXPORT Internal::Function function() const {return func_ref.function();}

    /** Return index to the function outputs. */
    EXPORT int index() const {return idx;}
};

namespace Internal {
struct ErrorBuffer;
class IRMutator;
}

/** A halide function. This class represents one stage in a Halide
 * pipeline, and is the unit by which we schedule things. By default
 * they are aggressively inlined, so you are encouraged to make lots
 * of little functions, rather than storing things in Exprs. */
class Func {

    /** A handle on the internal halide function that this
     * represents */
    Internal::Function func;

    /** When you make a reference to this function with fewer
     * arguments than it has dimensions, the argument list is bulked
     * up with 'implicit' vars with canonical names. This lets you
     * pass around partially applied Halide functions. */
    // @{
    std::pair<int, int> add_implicit_vars(std::vector<Var> &) const;
    std::pair<int, int> add_implicit_vars(std::vector<Expr> &) const;
    // @}

    /** The imaging pipeline that outputs this Func alone. */
    Pipeline pipeline_;

    /** Get the imaging pipeline that outputs this Func alone,
     * creating it (and freezing the Func) if necessary. */
    Pipeline pipeline();

    // Helper function for recursive reordering support
    EXPORT Func &reorder_storage(const std::vector<Var> &dims, size_t start);

    EXPORT void invalidate_cache();

public:

    /** Declare a new undefined function with the given name */
    EXPORT explicit Func(const std::string &name);

    /** Declare a new undefined function with an
     * automatically-generated unique name */
    EXPORT Func();

    /** Declare a new function with an automatically-generated unique
     * name, and define it to return the given expression (which may
     * not contain free variables). */
    EXPORT explicit Func(Expr e);

    /** Construct a new Func to wrap an existing, already-define
     * Function object. */
    EXPORT explicit Func(Internal::Function f);

    /** Construct a new Func to wrap a Buffer. */
    template<typename T>
    NO_INLINE explicit Func(Buffer<T> &im) : Func() {
        (*this)(_) = im(_);
    }

    /** Evaluate this function over some rectangular domain and return
     * the resulting buffer or buffers. Performs compilation if the
     * Func has not previously been realized and jit_compile has not
     * been called. If the final stage of the pipeline is on the GPU,
     * data is copied back to the host before being returned. The
     * returned Realization should probably be instantly converted to
     * a Buffer class of the appropriate type. That is, do this:
     *
     \code
     f(x) = sin(x);
     Buffer<float> im = f.realize(...);
     \endcode
     *
     * If your Func has multiple values, because you defined it using
     * a Tuple, then casting the result of a realize call to a buffer
     * or image will produce a run-time error. Instead you should do the
     * following:
     *
     \code
     f(x) = Tuple(x, sin(x));
     Realization r = f.realize(...);
     Buffer<int> im0 = r[0];
     Buffer<float> im1 = r[1];
     \endcode
     *
     */
    // @{
    EXPORT Realization realize(std::vector<int32_t> sizes, const Target &target = Target());
    EXPORT Realization realize(int x_size, int y_size, int z_size, int w_size,
                               const Target &target = Target());
    EXPORT Realization realize(int x_size, int y_size, int z_size,
                               const Target &target = Target());
    EXPORT Realization realize(int x_size, int y_size,
                               const Target &target = Target());
    EXPORT Realization realize(int x_size,
                               const Target &target = Target());
    EXPORT Realization realize(const Target &target = Target());
    // @}

    /** Evaluate this function into an existing allocated buffer or
     * buffers. If the buffer is also one of the arguments to the
     * function, strange things may happen, as the pipeline isn't
     * necessarily safe to run in-place. If you pass multiple buffers,
     * they must have matching sizes. This form of realize does *not*
     * automatically copy data back from the GPU. */
    EXPORT void realize(Realization dst, const Target &target = Target());

    /** For a given size of output, or a given output buffer,
     * determine the bounds required of all unbound ImageParams
     * referenced. Communicates the result by allocating new buffers
     * of the appropriate size and binding them to the unbound
     * ImageParams. */
    // @{
    EXPORT void infer_input_bounds(int x_size = 0, int y_size = 0, int z_size = 0, int w_size = 0);
    EXPORT void infer_input_bounds(Realization dst);
    // @}

    /** Statically compile this function to llvm bitcode, with the
     * given filename (which should probably end in .bc), type
     * signature, and C function name (which defaults to the same name
     * as this halide function */
    //@{
    EXPORT void compile_to_bitcode(const std::string &filename, const std::vector<Argument> &, const std::string &fn_name,
                                   const Target &target = get_target_from_environment());
    EXPORT void compile_to_bitcode(const std::string &filename, const std::vector<Argument> &,
                                   const Target &target = get_target_from_environment());
    // @}

    /** Statically compile this function to llvm assembly, with the
     * given filename (which should probably end in .ll), type
     * signature, and C function name (which defaults to the same name
     * as this halide function */
    //@{
    EXPORT void compile_to_llvm_assembly(const std::string &filename, const std::vector<Argument> &, const std::string &fn_name,
                                         const Target &target = get_target_from_environment());
    EXPORT void compile_to_llvm_assembly(const std::string &filename, const std::vector<Argument> &,
                                         const Target &target = get_target_from_environment());
    // @}

    /** Statically compile this function to an object file, with the
     * given filename (which should probably end in .o or .obj), type
     * signature, and C function name (which defaults to the same name
     * as this halide function. You probably don't want to use this
     * directly; call compile_to_static_library or compile_to_file instead. */
    //@{
    EXPORT void compile_to_object(const std::string &filename, const std::vector<Argument> &, const std::string &fn_name,
                                  const Target &target = get_target_from_environment());
    EXPORT void compile_to_object(const std::string &filename, const std::vector<Argument> &,
                                  const Target &target = get_target_from_environment());
    // @}

    /** Emit a header file with the given filename for this
     * function. The header will define a function with the type
     * signature given by the second argument, and a name given by the
     * third. The name defaults to the same name as this halide
     * function. You don't actually have to have defined this function
     * yet to call this. You probably don't want to use this directly;
     * call compile_to_static_library or compile_to_file instead. */
    EXPORT void compile_to_header(const std::string &filename, const std::vector<Argument> &, const std::string &fn_name = "",
                                  const Target &target = get_target_from_environment());

    /** Statically compile this function to text assembly equivalent
     * to the object file generated by compile_to_object. This is
     * useful for checking what Halide is producing without having to
     * disassemble anything, or if you need to feed the assembly into
     * some custom toolchain to produce an object file (e.g. iOS) */
    //@{
    EXPORT void compile_to_assembly(const std::string &filename, const std::vector<Argument> &, const std::string &fn_name,
                                    const Target &target = get_target_from_environment());
    EXPORT void compile_to_assembly(const std::string &filename, const std::vector<Argument> &,
                                    const Target &target = get_target_from_environment());
    // @}

    /** Statically compile this function to C source code. This is
     * useful for providing fallback code paths that will compile on
     * many platforms. Vectorization will fail, and parallelization
     * will produce serial code. */
    EXPORT void compile_to_c(const std::string &filename,
                             const std::vector<Argument> &,
                             const std::string &fn_name = "",
                             const Target &target = get_target_from_environment());

    /** Statically compile this function to C++ Tiramisu source code. */
    EXPORT void compile_to_tiramisu(const std::string &filename,
                                    const std::string &fn_name = "",
                                    const Target &target = get_target_from_environment());

    /** Write out an internal representation of lowered code. Useful
     * for analyzing and debugging scheduling. Can emit html or plain
     * text. */
    EXPORT void compile_to_lowered_stmt(const std::string &filename,
                                        const std::vector<Argument> &args,
                                        StmtOutputFormat fmt = Text,
                                        const Target &target = get_target_from_environment());

    /** Write out the loop nests specified by the schedule for this
     * Function. Helpful for understanding what a schedule is
     * doing. */
    EXPORT void print_loop_nest();

    /** Compile to object file and header pair, with the given
     * arguments. The name defaults to the same name as this halide
     * function.
     */
    EXPORT void compile_to_file(const std::string &filename_prefix, const std::vector<Argument> &args,
                                const std::string &fn_name = "",
                                const Target &target = get_target_from_environment());

    /** Compile to static-library file and header pair, with the given
     * arguments. The name defaults to the same name as this halide
     * function.
     */
    EXPORT void compile_to_static_library(const std::string &filename_prefix, const std::vector<Argument> &args,
                                          const std::string &fn_name = "",
                                          const Target &target = get_target_from_environment());

    /** Compile to static-library file and header pair once for each target;
     * each resulting function will be considered (in order) via halide_can_use_target_features()
     * at runtime, with the first appropriate match being selected for subsequent use.
     * This is typically useful for specializations that may vary unpredictably by machine
     * (e.g., SSE4.1/AVX/AVX2 on x86 desktop machines).
     * All targets must have identical arch-os-bits.
     */
    EXPORT void compile_to_multitarget_static_library(const std::string &filename_prefix,
                                                      const std::vector<Argument> &args,
                                                      const std::vector<Target> &targets);

    /** Store an internal representation of lowered code as a self
     * contained Module suitable for further compilation. */
    EXPORT Module compile_to_module(const std::vector<Argument> &args, const std::string &fn_name = "",
                                    const Target &target = get_target_from_environment());

    /** Compile and generate multiple target files with single call.
     * Deduces target files based on filenames specified in
     * output_files struct.
     */
    EXPORT void compile_to(const Outputs &output_files,
                           const std::vector<Argument> &args,
                           const std::string &fn_name,
                           const Target &target = get_target_from_environment());

    /** Eagerly jit compile the function to machine code. This
     * normally happens on the first call to realize. If you're
     * running your halide pipeline inside time-sensitive code and
     * wish to avoid including the time taken to compile a pipeline,
     * then you can call this ahead of time. Returns the raw function
     * pointer to the compiled pipeline. Default is to use the Target
     * returned from Halide::get_jit_target_from_environment()
     */
    EXPORT void *compile_jit(const Target &target = get_jit_target_from_environment());

    /** Set the error handler function that be called in the case of
     * runtime errors during halide pipelines. If you are compiling
     * statically, you can also just define your own function with
     * signature
     \code
     extern "C" void halide_error(void *user_context, const char *);
     \endcode
     * This will clobber Halide's version.
     */
    EXPORT void set_error_handler(void (*handler)(void *, const char *));

    /** Set a custom malloc and free for halide to use. Malloc should
     * return 32-byte aligned chunks of memory, and it should be safe
     * for Halide to read slightly out of bounds (up to 8 bytes before
     * the start or beyond the end). If compiling statically, routines
     * with appropriate signatures can be provided directly
    \code
     extern "C" void *halide_malloc(void *, size_t)
     extern "C" void halide_free(void *, void *)
     \endcode
     * These will clobber Halide's versions. See \file HalideRuntime.h
     * for declarations.
     */
    EXPORT void set_custom_allocator(void *(*malloc)(void *, size_t),
                                     void (*free)(void *, void *));

    /** Set a custom task handler to be called by the parallel for
     * loop. It is useful to set this if you want to do some
     * additional bookkeeping at the granularity of parallel
     * tasks. The default implementation does this:
     \code
     extern "C" int halide_do_task(void *user_context,
                                   int (*f)(void *, int, uint8_t *),
                                   int idx, uint8_t *state) {
         return f(user_context, idx, state);
     }
     \endcode
     * If you are statically compiling, you can also just define your
     * own version of the above function, and it will clobber Halide's
     * version.
     *
     * If you're trying to use a custom parallel runtime, you probably
     * don't want to call this. See instead \ref Func::set_custom_do_par_for .
    */
    EXPORT void set_custom_do_task(
        int (*custom_do_task)(void *, int (*)(void *, int, uint8_t *),
                              int, uint8_t *));

    /** Set a custom parallel for loop launcher. Useful if your app
     * already manages a thread pool. The default implementation is
     * equivalent to this:
     \code
     extern "C" int halide_do_par_for(void *user_context,
                                      int (*f)(void *, int, uint8_t *),
                                      int min, int extent, uint8_t *state) {
         int exit_status = 0;
         parallel for (int idx = min; idx < min+extent; idx++) {
             int job_status = halide_do_task(user_context, f, idx, state);
             if (job_status) exit_status = job_status;
         }
         return exit_status;
     }
     \endcode
     *
     * However, notwithstanding the above example code, if one task
     * fails, we may skip over other tasks, and if two tasks return
     * different error codes, we may select one arbitrarily to return.
     *
     * If you are statically compiling, you can also just define your
     * own version of the above function, and it will clobber Halide's
     * version.
     */
    EXPORT void set_custom_do_par_for(
        int (*custom_do_par_for)(void *, int (*)(void *, int, uint8_t *), int,
                                 int, uint8_t *));

    /** Set custom routines to call when tracing is enabled. Call this
     * on the output Func of your pipeline. This then sets custom
     * routines for the entire pipeline, not just calls to this
     * Func.
     *
     * If you are statically compiling, you can also just define your
     * own versions of the tracing functions (see HalideRuntime.h),
     * and they will clobber Halide's versions. */
    EXPORT void set_custom_trace(int (*trace_fn)(void *, const halide_trace_event_t *));

    /** Set the function called to print messages from the runtime.
     * If you are compiling statically, you can also just define your
     * own function with signature
     \code
     extern "C" void halide_print(void *user_context, const char *);
     \endcode
     * This will clobber Halide's version.
     */
    EXPORT void set_custom_print(void (*handler)(void *, const char *));

    /** Get a struct containing the currently set custom functions
     * used by JIT. */
    EXPORT const Internal::JITHandlers &jit_handlers();

    /** Add a custom pass to be used during lowering. It is run after
     * all other lowering passes. Can be used to verify properties of
     * the lowered Stmt, instrument it with extra code, or otherwise
     * modify it. The Func takes ownership of the pass, and will call
     * delete on it when the Func goes out of scope. So don't pass a
     * stack object, or share pass instances between multiple
     * Funcs. */
    template<typename T>
    void add_custom_lowering_pass(T *pass) {
        // Template instantiate a custom deleter for this type, then
        // cast it to a deleter that takes a IRMutator *. The custom
        // deleter lives in user code, so that deletion is on the same
        // heap as construction (I hate Windows).
        void (*deleter)(Internal::IRMutator *) =
            (void (*)(Internal::IRMutator *))(&delete_lowering_pass<T>);
        add_custom_lowering_pass(pass, deleter);
    }

    /** Add a custom pass to be used during lowering, with the
     * function that will be called to delete it also passed in. Set
     * it to nullptr if you wish to retain ownership of the object. */
    EXPORT void add_custom_lowering_pass(Internal::IRMutator *pass, void (*deleter)(Internal::IRMutator *));

    /** Remove all previously-set custom lowering passes */
    EXPORT void clear_custom_lowering_passes();

    /** Get the custom lowering passes. */
    EXPORT const std::vector<CustomLoweringPass> &custom_lowering_passes();

    /** When this function is compiled, include code that dumps its
     * values to a file after it is realized, for the purpose of
     * debugging.
     *
     * If filename ends in ".tif" or ".tiff" (case insensitive) the file
     * is in TIFF format and can be read by standard tools. Oherwise, the
     * file format is as follows:
     *
     * All data is in the byte-order of the target platform.  First, a
     * 20 byte-header containing four 32-bit ints, giving the extents
     * of the first four dimensions.  Dimensions beyond four are
     * folded into the fourth.  Then, a fifth 32-bit int giving the
     * data type of the function. The typecodes are given by: float =
     * 0, double = 1, uint8_t = 2, int8_t = 3, uint16_t = 4, int16_t =
     * 5, uint32_t = 6, int32_t = 7, uint64_t = 8, int64_t = 9. The
     * data follows the header, as a densely packed array of the given
     * size and the given type. If given the extension .tmp, this file
     * format can be natively read by the program ImageStack. */
    EXPORT void debug_to_file(const std::string &filename);

    /** The name of this function, either given during construction,
     * or automatically generated. */
    EXPORT const std::string &name() const;

    /** Get the pure arguments. */
    EXPORT std::vector<Var> args() const;

    /** The right-hand-side value of the pure definition of this
     * function. Causes an error if there's no pure definition, or if
     * the function is defined to return multiple values. */
    EXPORT Expr value() const;

    /** The values returned by this function. An error if the function
     * has not been been defined. Returns a Tuple with one element for
     * functions defined to return a single value. */
    EXPORT Tuple values() const;

    /** Does this function have at least a pure definition. */
    EXPORT bool defined() const;

    /** Get the left-hand-side of the update definition. An empty
     * vector if there's no update definition. If there are
     * multiple update definitions for this function, use the
     * argument to select which one you want. */
    EXPORT const std::vector<Expr> &update_args(int idx = 0) const;

    /** Get the right-hand-side of an update definition. An error if
     * there's no update definition. If there are multiple
     * update definitions for this function, use the argument to
     * select which one you want. */
    EXPORT Expr update_value(int idx = 0) const;

    /** Get the right-hand-side of an update definition for
     * functions that returns multiple values. An error if there's no
     * update definition. Returns a Tuple with one element for
     * functions that return a single value. */
    EXPORT Tuple update_values(int idx = 0) const;

    /** Get the RVars of the reduction domain for an update definition, if there is
     * one. */
    EXPORT std::vector<RVar> rvars(int idx = 0) const;

    /** Does this function have at least one update definition? */
    EXPORT bool has_update_definition() const;

    /** How many update definitions does this function have? */
    EXPORT int num_update_definitions() const;

    /** Is this function an external stage? That is, was it defined
     * using define_extern? */
    EXPORT bool is_extern() const;

    /** Add an extern definition for this Func. This lets you define a
     * Func that represents an external pipeline stage. You can, for
     * example, use it to wrap a call to an extern library such as
     * fftw. */
    // @{
    EXPORT void define_extern(const std::string &function_name,
                              const std::vector<ExternFuncArgument> &params,
                              Type t,
                              int dimensionality,
                              NameMangling mangling,
                              bool uses_old_buffer_t) {
        define_extern(function_name, params, std::vector<Type>{t},
                      dimensionality, mangling, DeviceAPI::Host, uses_old_buffer_t);
    }

    EXPORT void define_extern(const std::string &function_name,
                              const std::vector<ExternFuncArgument> &params,
                              Type t,
                              int dimensionality,
                              NameMangling mangling = NameMangling::Default,
                              DeviceAPI device_api = DeviceAPI::Host,
                              bool uses_old_buffer_t = false) {
        define_extern(function_name, params, std::vector<Type>{t},
                      dimensionality, mangling, device_api, uses_old_buffer_t);
    }

    EXPORT void define_extern(const std::string &function_name,
                              const std::vector<ExternFuncArgument> &params,
                              const std::vector<Type> &types,
                              int dimensionality,
                              NameMangling mangling,
                              bool uses_old_buffer_t) {
      define_extern(function_name, params, types,
                    dimensionality, mangling, DeviceAPI::Host, uses_old_buffer_t);
    }

    EXPORT void define_extern(const std::string &function_name,
                              const std::vector<ExternFuncArgument> &params,
                              const std::vector<Type> &types,
                              int dimensionality,
                              NameMangling mangling = NameMangling::Default,
                              DeviceAPI device_api = DeviceAPI::Host,
                              bool uses_old_buffer_t = false);
    // @}

    /** Get the types of the outputs of this Func. */
    EXPORT const std::vector<Type> &output_types() const;

    /** Get the number of outputs of this Func. Corresponds to the
     * size of the Tuple this Func was defined to return. */
    EXPORT int outputs() const;

    /** Get the name of the extern function called for an extern
     * definition. */
    EXPORT const std::string &extern_function_name() const;

    /** The dimensionality (number of arguments) of this
     * function. Zero if the function is not yet defined. */
    EXPORT int dimensions() const;

    /** Construct either the left-hand-side of a definition, or a call
     * to a functions that happens to only contain vars as
     * arguments. If the function has already been defined, and fewer
     * arguments are given than the function has dimensions, then
     * enough implicit vars are added to the end of the argument list
     * to make up the difference (see \ref Var::implicit) */
    // @{
    EXPORT FuncRef operator()(std::vector<Var>) const;

    template <typename... Args>
    NO_INLINE typename std::enable_if<Internal::all_are_convertible<Var, Args...>::value, FuncRef>::type
    operator()(Args&&... args) const {
        std::vector<Var> collected_args{std::forward<Args>(args)...};
        return this->operator()(collected_args);
    }
    // @}

    /** Either calls to the function, or the left-hand-side of
     * an update definition (see \ref RDom). If the function has
     * already been defined, and fewer arguments are given than the
     * function has dimensions, then enough implicit vars are added to
     * the end of the argument list to make up the difference. (see
     * \ref Var::implicit)*/
    // @{
    EXPORT FuncRef operator()(std::vector<Expr>) const;

    template <typename... Args>
    NO_INLINE typename std::enable_if<Internal::all_are_convertible<Expr, Args...>::value, FuncRef>::type
    operator()(Expr x, Args&&... args) const {
        std::vector<Expr> collected_args{x, std::forward<Args>(args)...};
        return (*this)(collected_args);
    }
    // @}

    /** Creates and returns a new identity Func that wraps this Func. During
     * compilation, Halide replaces all calls to this Func done by 'f'
     * with calls to the wrapper. If this Func is already wrapped for
     * use in 'f', will return the existing wrapper.
     *
     * For example, g.in(f) would rewrite a pipeline like this:
     \code
     g(x, y) = ...
     f(x, y) = ... g(x, y) ...
     \endcode
     * into a pipeline like this:
     \code
     g(x, y) = ...
     g_wrap(x, y) = g(x, y)
     f(x, y) = ... g_wrap(x, y)
     \endcode
     *
     * This has a variety of uses. You can use it to schedule this
     * Func differently in the different places it is used:
     \code
     g(x, y) = ...
     f1(x, y) = ... g(x, y) ...
     f2(x, y) = ... g(x, y) ...
     g.in(f1).compute_at(f1, y).vectorize(x, 8);
     g.in(f2).compute_at(f2, x).unroll(x);
     \endcode
     *
     * You can also use it to stage loads from this Func via some
     * intermediate buffer (perhaps on the stack as in
     * test/performance/block_transpose.cpp, or in shared GPU memory
     * as in test/performance/wrap.cpp). In this we compute the
     * wrapper at tiles of the consuming Funcs like so:
     \code
     g.compute_root()...
     g.in(f).compute_at(f, tiles)...
     \endcode
     *
     * Func::in() can also be used to compute pieces of a Func into a
     * smaller scratch buffer (perhaps on the GPU) and then copy them
     * into a larger output buffer one tile at a time. See
     * apps/interpolate/interpolate.cpp for an example of this. In
     * this case we compute the Func at tiles of its own wrapper:
     \code
     f.in(g).compute_root().gpu_tile(...)...
     f.compute_at(f.in(g), tiles)...
     \endcode
     *
     * A similar use of Func::in() wrapping Funcs with multiple update
     * stages in a pure wrapper. The following code:
     \code
     f(x, y) = x + y;
     f(x, y) += 5;
     g(x, y) = f(x, y);
     f.compute_root();
     \endcode
     *
     * Is equivalent to:
     \code
     for y:
       for x:
         f(x, y) = x + y;
     for y:
       for x:
         f(x, y) += 5
     for y:
       for x:
         g(x, y) = f(x, y)
     \endcode
     * using Func::in(), we can write:
     \code
     f(x, y) = x + y;
     f(x, y) += 5;
     g(x, y) = f(x, y);
     f.in(g).compute_root();
     \endcode
     * which instead produces:
     \code
     for y:
       for x:
         f(x, y) = x + y;
         f(x, y) += 5
         f_wrap(x, y) = f(x, y)
     for y:
       for x:
         g(x, y) = f_wrap(x, y)
     \endcode
     */
    EXPORT Func in(const Func &f);

    /** Create and return an identity wrapper shared by all the Funcs in
     * 'fs'. If any of the Funcs in 'fs' already have a custom wrapper,
     * this will throw an error. */
    EXPORT Func in(const std::vector<Func> &fs);

    /** Create and return a global identity wrapper, which wraps all calls to
     * this Func by any other Func. If a global wrapper already exists,
     * returns it. The global identity wrapper is only used by callers for
     * which no custom wrapper has been specified.
     */
    EXPORT Func in();

    /** Similar to \ref Func::in; however, instead of replacing the call to
     * this Func with an identity Func that refers to it, this replaces the
     * call with a clone of this Func.
     *
     * For example, f.clone_in(g) would rewrite a pipeline like this:
     \code
     f(x, y) = x + y;
     g(x, y) = f(x, y) + 2;
     h(x, y) = f(x, y) - 3;
     \endcode
     * into a pipeline like this:
     \code
     f(x, y) = x + y;
     f_clone(x, y) = x + y;
     g(x, y) = f_clone(x, y) + 2;
     h(x, y) = f(x, y) - 3;
     \endcode
     *
     */
    //@{
    EXPORT Func clone_in(const Func &f);
    EXPORT Func clone_in(const std::vector<Func> &fs);
    //@}

    /** Declare that this function should be implemented by a call to
     * halide_buffer_copy with the given target device API. Asserts
     * that the Func has a pure definition which is a simple call to a
     * single input, and no update definitions. The wrapper Funcs
     * returned by in() are suitable candidates. Consumes all pure
     * variables, and rewrites the Func to have an extern definition
     * that calls halide_buffer_copy. */
    EXPORT Func copy_to_device(DeviceAPI d = DeviceAPI::Default_GPU);

    /** Declare that this function should be implemented by a call to
     * halide_buffer_copy with a NULL target device API. Equivalent to
     * copy_to_device(DeviceAPI::Host). Asserts that the Func has a
     * pure definition which is a simple call to a single input, and
     * no update definitions. The wrapper Funcs returned by in() are
     * suitable candidates. Consumes all pure variables, and rewrites
     * the Func to have an extern definition that calls
     * halide_buffer_copy.
     *
     * Note that if the source Func is already valid in host memory,
     * this compiles to code that does the minimum number of calls to
     * memcpy.
     */
    EXPORT Func copy_to_host();

    /** Split a dimension into inner and outer subdimensions with the
     * given names, where the inner dimension iterates from 0 to
     * factor-1. The inner and outer subdimensions can then be dealt
     * with using the other scheduling calls. It's ok to reuse the old
     * variable name as either the inner or outer variable. The final
     * argument specifies how the tail should be handled if the split
     * factor does not provably divide the extent. */
    EXPORT Func &split(VarOrRVar old, VarOrRVar outer, VarOrRVar inner, Expr factor, TailStrategy tail = TailStrategy::Auto);

    /** Join two dimensions into a single fused dimenion. The fused
     * dimension covers the product of the extents of the inner and
     * outer dimensions given. */
    EXPORT Func &fuse(VarOrRVar inner, VarOrRVar outer, VarOrRVar fused);

    /** Mark a dimension to be traversed serially. This is the default. */
    EXPORT Func &serial(VarOrRVar var);

    /** Mark a dimension to be traversed in parallel */
    EXPORT Func &parallel(VarOrRVar var);

    /** Split a dimension by the given task_size, and the parallelize the
     * outer dimension. This creates parallel tasks that have size
     * task_size. After this call, var refers to the outer dimension of
     * the split. The inner dimension has a new anonymous name. If you
     * wish to mutate it, or schedule with respect to it, do the split
     * manually. */
    EXPORT Func &parallel(VarOrRVar var, Expr task_size, TailStrategy tail = TailStrategy::Auto);

    /** Mark a dimension to be computed all-at-once as a single
     * vector. The dimension should have constant extent -
     * e.g. because it is the inner dimension following a split by a
     * constant factor. For most uses of vectorize you want the two
     * argument form. The variable to be vectorized should be the
     * innermost one. */
    EXPORT Func &vectorize(VarOrRVar var);

    /** Mark a dimension to be completely unrolled. The dimension
     * should have constant extent - e.g. because it is the inner
     * dimension following a split by a constant factor. For most uses
     * of unroll you want the two-argument form. */
    EXPORT Func &unroll(VarOrRVar var);

    /** Split a dimension by the given factor, then vectorize the
     * inner dimension. This is how you vectorize a loop of unknown
     * size. The variable to be vectorized should be the innermost
     * one. After this call, var refers to the outer dimension of the
     * split. 'factor' must be an integer. */
    EXPORT Func &vectorize(VarOrRVar var, Expr factor, TailStrategy tail = TailStrategy::Auto);

    /** Split a dimension by the given factor, then unroll the inner
     * dimension. This is how you unroll a loop of unknown size by
     * some constant factor. After this call, var refers to the outer
     * dimension of the split. 'factor' must be an integer. */
    EXPORT Func &unroll(VarOrRVar var, Expr factor, TailStrategy tail = TailStrategy::Auto);

    /** Statically declare that the range over which a function should
     * be evaluated is given by the second and third arguments. This
     * can let Halide perform some optimizations. E.g. if you know
     * there are going to be 4 color channels, you can completely
     * vectorize the color channel dimension without the overhead of
     * splitting it up. If bounds inference decides that it requires
     * more of this function than the bounds you have stated, a
     * runtime error will occur when you try to run your pipeline. */
    EXPORT Func &bound(Var var, Expr min, Expr extent);

    /** Statically declare the range over which the function will be
     * evaluated in the general case. This provides a basis for the auto
     * scheduler to make trade-offs and scheduling decisions. The auto
     * generated schedules might break when the sizes of the dimensions are
     * very different from the estimates specified. These estimates are used
     * only by the auto scheduler if the function is a pipeline output. */
    EXPORT Func &estimate(Var var, Expr min, Expr extent);

    /** Expand the region computed so that the min coordinates is
     * congruent to 'remainder' modulo 'modulus', and the extent is a
     * multiple of 'modulus'. For example, f.align_bounds(x, 2) forces
     * the min and extent realized to be even, and calling
     * f.align_bounds(x, 2, 1) forces the min to be odd and the extent
     * to be even. The region computed always contains the region that
     * would have been computed without this directive, so no
     * assertions are injected. */
    EXPORT Func &align_bounds(Var var, Expr modulus, Expr remainder = 0);

    /** Bound the extent of a Func's realization, but not its
     * min. This means the dimension can be unrolled or vectorized
     * even when its min is not fixed (for example because it is
     * compute_at tiles of another Func). This can also be useful for
     * forcing a function's allocation to be a fixed size, which often
     * means it can go on the stack. */
    EXPORT Func &bound_extent(Var var, Expr extent);

    /** Split two dimensions at once by the given factors, and then
     * reorder the resulting dimensions to be xi, yi, xo, yo from
     * innermost outwards. This gives a tiled traversal. */
    EXPORT Func &tile(VarOrRVar x, VarOrRVar y,
                      VarOrRVar xo, VarOrRVar yo,
                      VarOrRVar xi, VarOrRVar yi,
                      Expr xfactor, Expr yfactor,
                      TailStrategy tail = TailStrategy::Auto);

    /** A shorter form of tile, which reuses the old variable names as
     * the new outer dimensions */
    EXPORT Func &tile(VarOrRVar x, VarOrRVar y,
                      VarOrRVar xi, VarOrRVar yi,
                      Expr xfactor, Expr yfactor,
                      TailStrategy tail = TailStrategy::Auto);

    /** Reorder variables to have the given nesting order, from
     * innermost out */
    EXPORT Func &reorder(const std::vector<VarOrRVar> &vars);

    template <typename... Args>
    NO_INLINE typename std::enable_if<Internal::all_are_convertible<VarOrRVar, Args...>::value, Func &>::type
    reorder(VarOrRVar x, VarOrRVar y, Args&&... args) {
        std::vector<VarOrRVar> collected_args{x, y, std::forward<Args>(args)...};
        return reorder(collected_args);
    }

    /** Rename a dimension. Equivalent to split with a inner size of one. */
    EXPORT Func &rename(VarOrRVar old_name, VarOrRVar new_name);

    /** Specify that race conditions are permitted for this Func,
     * which enables parallelizing over RVars even when Halide cannot
     * prove that it is safe to do so. Use this with great caution,
     * and only if you can prove to yourself that this is safe, as it
     * may result in a non-deterministic routine that returns
     * different values at different times or on different machines. */
    EXPORT Func &allow_race_conditions();


    /** Specialize a Func. This creates a special-case version of the
     * Func where the given condition is true. The most effective
     * conditions are those of the form param == value, and boolean
     * Params. Consider a simple example:
     \code
     f(x) = x + select(cond, 0, 1);
     f.compute_root();
     \endcode
     * This is equivalent to:
     \code
     for (int x = 0; x < width; x++) {
       f[x] = x + (cond ? 0 : 1);
     }
     \endcode
     * Adding the scheduling directive:
     \code
     f.specialize(cond)
     \endcode
     * makes it equivalent to:
     \code
     if (cond) {
       for (int x = 0; x < width; x++) {
         f[x] = x;
       }
     } else {
       for (int x = 0; x < width; x++) {
         f[x] = x + 1;
       }
     }
     \endcode
     * Note that the inner loops have been simplified. In the first
     * path Halide knows that cond is true, and in the second path
     * Halide knows that it is false.
     *
     * The specialized version gets its own schedule, which inherits
     * every directive made about the parent Func's schedule so far
     * except for its specializations. This method returns a handle to
     * the new schedule. If you wish to retrieve the specialized
     * sub-schedule again later, you can call this method with the
     * same condition. Consider the following example of scheduling
     * the specialized version:
     *
     \code
     f(x) = x;
     f.compute_root();
     f.specialize(width > 1).unroll(x, 2);
     \endcode
     * Assuming for simplicity that width is even, this is equivalent to:
     \code
     if (width > 1) {
       for (int x = 0; x < width/2; x++) {
         f[2*x] = 2*x;
         f[2*x + 1] = 2*x + 1;
       }
     } else {
       for (int x = 0; x < width/2; x++) {
         f[x] = x;
       }
     }
     \endcode
     * For this case, it may be better to schedule the un-specialized
     * case instead:
     \code
     f(x) = x;
     f.compute_root();
     f.specialize(width == 1); // Creates a copy of the schedule so far.
     f.unroll(x, 2); // Only applies to the unspecialized case.
     \endcode
     * This is equivalent to:
     \code
     if (width == 1) {
       f[0] = 0;
     } else {
       for (int x = 0; x < width/2; x++) {
         f[2*x] = 2*x;
         f[2*x + 1] = 2*x + 1;
       }
     }
     \endcode
     * This can be a good way to write a pipeline that splits,
     * vectorizes, or tiles, but can still handle small inputs.
     *
     * If a Func has several specializations, the first matching one
     * will be used, so the order in which you define specializations
     * is significant. For example:
     *
     \code
     f(x) = x + select(cond1, a, b) - select(cond2, c, d);
     f.specialize(cond1);
     f.specialize(cond2);
     \endcode
     * is equivalent to:
     \code
     if (cond1) {
       for (int x = 0; x < width; x++) {
         f[x] = x + a - (cond2 ? c : d);
       }
     } else if (cond2) {
       for (int x = 0; x < width; x++) {
         f[x] = x + b - c;
       }
     } else {
       for (int x = 0; x < width; x++) {
         f[x] = x + b - d;
       }
     }
     \endcode
     *
     * Specializations may in turn be specialized, which creates a
     * nested if statement in the generated code.
     *
     \code
     f(x) = x + select(cond1, a, b) - select(cond2, c, d);
     f.specialize(cond1).specialize(cond2);
     \endcode
     * This is equivalent to:
     \code
     if (cond1) {
       if (cond2) {
         for (int x = 0; x < width; x++) {
           f[x] = x + a - c;
         }
       } else {
         for (int x = 0; x < width; x++) {
           f[x] = x + a - d;
         }
       }
     } else {
       for (int x = 0; x < width; x++) {
         f[x] = x + b - (cond2 ? c : d);
       }
     }
     \endcode
     * To create a 4-way if statement that simplifies away all of the
     * ternary operators above, you could say:
     \code
     f.specialize(cond1).specialize(cond2);
     f.specialize(cond2);
     \endcode
     * or
     \code
     f.specialize(cond1 && cond2);
     f.specialize(cond1);
     f.specialize(cond2);
     \endcode
     *
     * Any prior Func which is compute_at some variable of this Func
     * gets separately included in all paths of the generated if
     * statement. The Var in the compute_at call to must exist in all
     * paths, but it may have been generated via a different path of
     * splits, fuses, and renames. This can be used somewhat
     * creatively. Consider the following code:
     \code
     g(x, y) = 8*x;
     f(x, y) = g(x, y) + 1;
     f.compute_root().specialize(cond);
     Var g_loop;
     f.specialize(cond).rename(y, g_loop);
     f.rename(x, g_loop);
     g.compute_at(f, g_loop);
     \endcode
     * When cond is true, this is equivalent to g.compute_at(f,y).
     * When it is false, this is equivalent to g.compute_at(f,x).
     */
    EXPORT Stage specialize(Expr condition);

    /** Add a specialization to a Func that always terminates execution
     * with a call to halide_error(). By itself, this is of limited use,
     * but can be useful to terminate chains of specialize() calls where
     * no "default" case is expected (thus avoiding unnecessary code generation).
     *
     * For instance, say we want to optimize a pipeline to process images
     * in planar and interleaved format; we might typically do something like:
     \code
     ImageParam im(UInt(8), 3);
     Func f = do_something_with(im);
     f.specialize(im.dim(0).stride() == 1).vectorize(x, 8);  // planar
     f.specialize(im.dim(2).stride() == 1).reorder(c, x, y).vectorize(c);  // interleaved
     \endcode
     * This code will vectorize along rows for the planar case, and across pixel
     * components for the interleaved case... but there is an implicit "else"
     * for the unhandled cases, which generates unoptimized code. If we never
     * anticipate passing any other sort of images to this, we code streamline
     * our code by adding specialize_fail():
     \code
     ImageParam im(UInt(8), 3);
     Func f = do_something(im);
     f.specialize(im.dim(0).stride() == 1).vectorize(x, 8);  // planar
     f.specialize(im.dim(2).stride() == 1).reorder(c, x, y).vectorize(c);  // interleaved
     f.specialize_fail("Unhandled image format");
     \endcode
     * Conceptually, this produces codes like:
     \code
     if (im.dim(0).stride() == 1) {
        do_something_planar();
     } else if (im.dim(2).stride() == 1) {
        do_something_interleaved();
     } else {
        halide_error("Unhandled image format");
     }
     \endcode
     *
     * Note that calling specialize_fail() terminates the specialization chain
     * for a given Func; you cannot create new specializations for the Func
     * afterwards (though you can retrieve handles to previous specializations).
     */
    EXPORT void specialize_fail(const std::string &message);

    /** Tell Halide that the following dimensions correspond to GPU
     * thread indices. This is useful if you compute a producer
     * function within the block indices of a consumer function, and
     * want to control how that function's dimensions map to GPU
     * threads. If the selected target is not an appropriate GPU, this
     * just marks those dimensions as parallel. */
    // @{
    EXPORT Func &gpu_threads(VarOrRVar thread_x, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_threads(VarOrRVar thread_x, VarOrRVar thread_y, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_threads(VarOrRVar thread_x, VarOrRVar thread_y, VarOrRVar thread_z, DeviceAPI device_api = DeviceAPI::Default_GPU);
    // @}

    /** Tell Halide to run this stage using a single gpu thread and
     * block. This is not an efficient use of your GPU, but it can be
     * useful to avoid copy-back for intermediate update stages that
     * touch a very small part of your Func. */
    EXPORT Func &gpu_single_thread(DeviceAPI device_api = DeviceAPI::Default_GPU);

    /** Tell Halide that the following dimensions correspond to GPU
     * block indices. This is useful for scheduling stages that will
     * run serially within each GPU block. If the selected target is
     * not ptx, this just marks those dimensions as parallel. */
    // @{
    EXPORT Func &gpu_blocks(VarOrRVar block_x, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_blocks(VarOrRVar block_x, VarOrRVar block_y, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_blocks(VarOrRVar block_x, VarOrRVar block_y, VarOrRVar block_z, DeviceAPI device_api = DeviceAPI::Default_GPU);
    // @}

    /** Tell Halide that the following dimensions correspond to GPU
     * block indices and thread indices. If the selected target is not
     * ptx, these just mark the given dimensions as parallel. The
     * dimensions are consumed by this call, so do all other
     * unrolling, reordering, etc first. */
    // @{
    EXPORT Func &gpu(VarOrRVar block_x, VarOrRVar thread_x, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu(VarOrRVar block_x, VarOrRVar block_y,
                     VarOrRVar thread_x, VarOrRVar thread_y, DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu(VarOrRVar block_x, VarOrRVar block_y, VarOrRVar block_z,
                     VarOrRVar thread_x, VarOrRVar thread_y, VarOrRVar thread_z, DeviceAPI device_api = DeviceAPI::Default_GPU);
    // @}

    /** Short-hand for tiling a domain and mapping the tile indices
     * to GPU block indices and the coordinates within each tile to
     * GPU thread indices. Consumes the variables given, so do all
     * other scheduling first. */
    // @{
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar bx, Var tx, Expr x_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar bx, RVar tx, Expr x_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar tx, Expr x_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y,
                          VarOrRVar bx, VarOrRVar by,
                          VarOrRVar tx, VarOrRVar ty,
                          Expr x_size, Expr y_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y,
                          VarOrRVar tx, Var ty,
                          Expr x_size, Expr y_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y,
                          VarOrRVar tx, RVar ty,
                          Expr x_size, Expr y_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);

    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y, VarOrRVar z,
                          VarOrRVar bx, VarOrRVar by, VarOrRVar bz,
                          VarOrRVar tx, VarOrRVar ty, VarOrRVar tz,
                          Expr x_size, Expr y_size, Expr z_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y, VarOrRVar z,
                          VarOrRVar tx, VarOrRVar ty, VarOrRVar tz,
                          Expr x_size, Expr y_size, Expr z_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);

    HALIDE_ATTRIBUTE_DEPRECATED("This form of gpu_tile() is deprecated.")
    EXPORT Func &gpu_tile(VarOrRVar x, Expr x_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    HALIDE_ATTRIBUTE_DEPRECATED("This form of gpu_tile() is deprecated.")
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y, Expr x_size, Expr y_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    HALIDE_ATTRIBUTE_DEPRECATED("This form of gpu_tile() is deprecated.")
    EXPORT Func &gpu_tile(VarOrRVar x, VarOrRVar y, VarOrRVar z,
                          Expr x_size, Expr y_size, Expr z_size,
                          TailStrategy tail = TailStrategy::Auto,
                          DeviceAPI device_api = DeviceAPI::Default_GPU);
    // @}

    /** Schedule for execution using coordinate-based hardware api.
     * GLSL is an example of this. Conceptually, this is
     * similar to parallelization over 'x' and 'y' (since GLSL shaders compute
     * individual output pixels in parallel) and vectorization over 'c'
     * (since GLSL/RS implicitly vectorizes the color channel). */
    EXPORT Func &shader(Var x, Var y, Var c, DeviceAPI device_api);

    /** Schedule for execution as GLSL kernel. */
    EXPORT Func &glsl(Var x, Var y, Var c);

    /** Schedule for execution on Hexagon. When a loop is marked with
     * Hexagon, that loop is executed on a Hexagon DSP. */
    EXPORT Func &hexagon(VarOrRVar x = Var::outermost());

    /** Prefetch data written to or read from a Func or an ImageParam by a
     * subsequent loop iteration, at an optionally specified iteration offset.
     * 'var' specifies at which loop level the prefetch calls should be inserted.
     * The final argument specifies how prefetch of region outside bounds
     * should be handled.
     *
     * For example, consider this pipeline:
     \code
     Func f, g;
     Var x, y;
     f(x, y) = x + y;
     g(x, y) = 2 * f(x, y);
     \endcode
     *
     * The following schedule:
     \code
     f.compute_root();
     g.prefetch(f, x, 2, PrefetchBoundStrategy::NonFaulting);
     \endcode
     *
     * will inject prefetch call at the innermost loop of 'g' and generate
     * the following loop nest:
     * for y = ...
     *   for x = ...
     *     f(x, y) = x + y
     * for y = ..
     *   for x = ...
     *     prefetch(&f[x + 2, y], 1, 16);
     *     g(x, y) = 2 * f(x, y)
     */
    // @{
    EXPORT Func &prefetch(const Func &f, VarOrRVar var, Expr offset = 1,
                          PrefetchBoundStrategy strategy = PrefetchBoundStrategy::GuardWithIf);
    EXPORT Func &prefetch(const Internal::Parameter &param, VarOrRVar var, Expr offset = 1,
                          PrefetchBoundStrategy strategy = PrefetchBoundStrategy::GuardWithIf);
    template<typename T>
    Func &prefetch(const T &image, VarOrRVar var, Expr offset = 1,
                   PrefetchBoundStrategy strategy = PrefetchBoundStrategy::GuardWithIf) {
        return prefetch(image.parameter(), var, offset, strategy);
    }
    // @}

    /** Specify how the storage for the function is laid out. These
     * calls let you specify the nesting order of the dimensions. For
     * example, foo.reorder_storage(y, x) tells Halide to use
     * column-major storage for any realizations of foo, without
     * changing how you refer to foo in the code. You may want to do
     * this if you intend to vectorize across y. When representing
     * color images, foo.reorder_storage(c, x, y) specifies packed
     * storage (red, green, and blue values adjacent in memory), and
     * foo.reorder_storage(x, y, c) specifies planar storage (entire
     * red, green, and blue images one after the other in memory).
     *
     * If you leave out some dimensions, those remain in the same
     * positions in the nesting order while the specified variables
     * are reordered around them. */
    // @{
    EXPORT Func &reorder_storage(const std::vector<Var> &dims);

    EXPORT Func &reorder_storage(Var x, Var y);
    template <typename... Args>
    NO_INLINE typename std::enable_if<Internal::all_are_convertible<Var, Args...>::value, Func &>::type
    reorder_storage(Var x, Var y, Args&&... args) {
        std::vector<Var> collected_args{x, y, std::forward<Args>(args)...};
        return reorder_storage(collected_args);
    }
    // @}

    /** Pad the storage extent of a particular dimension of
     * realizations of this function up to be a multiple of the
     * specified alignment. This guarantees that the strides for the
     * dimensions stored outside of dim will be multiples of the
     * specified alignment, where the strides and alignment are
     * measured in numbers of elements.
     *
     * For example, to guarantee that a function foo(x, y, c)
     * representing an image has scanlines starting on offsets
     * aligned to multiples of 16, use foo.align_storage(x, 16). */
    EXPORT Func &align_storage(Var dim, Expr alignment);

    /** Store realizations of this function in a circular buffer of a
     * given extent. This is more efficient when the extent of the
     * circular buffer is a power of 2. If the fold factor is too
     * small, or the dimension is not accessed monotonically, the
     * pipeline will generate an error at runtime.
     *
     * The fold_forward option indicates that the new values of the
     * producer are accessed by the consumer in a monotonically
     * increasing order. Folding storage of producers is also
     * supported if the new values are accessed in a monotonically
     * decreasing order by setting fold_forward to false.
     *
     * For example, consider the pipeline:
     \code
     Func f, g;
     Var x, y;
     g(x, y) = x*y;
     f(x, y) = g(x, y) + g(x, y+1);
     \endcode
     *
     * If we schedule f like so:
     *
     \code
     g.compute_at(f, y).store_root().fold_storage(y, 2);
     \endcode
     *
     * Then g will be computed at each row of f and stored in a buffer
     * with an extent in y of 2, alternately storing each computed row
     * of g in row y=0 or y=1.
     */
    EXPORT Func &fold_storage(Var dim, Expr extent, bool fold_forward = true);

    /** Compute this function as needed for each unique value of the
     * given var for the given calling function f.
     *
     * For example, consider the simple pipeline:
     \code
     Func f, g;
     Var x, y;
     g(x, y) = x*y;
     f(x, y) = g(x, y) + g(x, y+1) + g(x+1, y) + g(x+1, y+1);
     \endcode
     *
     * If we schedule f like so:
     *
     \code
     g.compute_at(f, x);
     \endcode
     *
     * Then the C code equivalent to this pipeline will look like this
     *
     \code

     int f[height][width];
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             int g[2][2];
             g[0][0] = x*y;
             g[0][1] = (x+1)*y;
             g[1][0] = x*(y+1);
             g[1][1] = (x+1)*(y+1);
             f[y][x] = g[0][0] + g[1][0] + g[0][1] + g[1][1];
         }
     }

     \endcode
     *
     * The allocation and computation of g is within f's loop over x,
     * and enough of g is computed to satisfy all that f will need for
     * that iteration. This has excellent locality - values of g are
     * used as soon as they are computed, but it does redundant
     * work. Each value of g ends up getting computed four times. If
     * we instead schedule f like so:
     *
     \code
     g.compute_at(f, y);
     \endcode
     *
     * The equivalent C code is:
     *
     \code
     int f[height][width];
     for (int y = 0; y < height; y++) {
         int g[2][width+1];
         for (int x = 0; x < width; x++) {
             g[0][x] = x*y;
             g[1][x] = x*(y+1);
         }
         for (int x = 0; x < width; x++) {
             f[y][x] = g[0][x] + g[1][x] + g[0][x+1] + g[1][x+1];
         }
     }
     \endcode
     *
     * The allocation and computation of g is within f's loop over y,
     * and enough of g is computed to satisfy all that f will need for
     * that iteration. This does less redundant work (each point in g
     * ends up being evaluated twice), but the locality is not quite
     * as good, and we have to allocate more temporary memory to store
     * g.
     */
    EXPORT Func &compute_at(Func f, Var var);

    /** Schedule a function to be computed within the iteration over
     * some dimension of an update domain. Produces equivalent code
     * to the version of compute_at that takes a Var. */
    EXPORT Func &compute_at(Func f, RVar var);

    /** Schedule a function to be computed within the iteration over
     * a given LoopLevel. */
    EXPORT Func &compute_at(LoopLevel loop_level);

    /** Compute all of this function once ahead of time. Reusing
     * the example in \ref Func::compute_at:
     *
     \code
     Func f, g;
     Var x, y;
     g(x, y) = x*y;
     f(x, y) = g(x, y) + g(x, y+1) + g(x+1, y) + g(x+1, y+1);

     g.compute_root();
     \endcode
     *
     * is equivalent to
     *
     \code
     int f[height][width];
     int g[height+1][width+1];
     for (int y = 0; y < height+1; y++) {
         for (int x = 0; x < width+1; x++) {
             g[y][x] = x*y;
         }
     }
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             f[y][x] = g[y][x] + g[y+1][x] + g[y][x+1] + g[y+1][x+1];
         }
     }
     \endcode
     *
     * g is computed once ahead of time, and enough is computed to
     * satisfy all uses of it. This does no redundant work (each point
     * in g is evaluated once), but has poor locality (values of g are
     * probably not still in cache when they are used by f), and
     * allocates lots of temporary memory to store g.
     */
    EXPORT Func &compute_root();

    /** Use the halide_memoization_cache_... interface to store a
     *  computed version of this function across invocations of the
     *  Func.
     */
    EXPORT Func &memoize();


    /** Allocate storage for this function within f's loop over
     * var. Scheduling storage is optional, and can be used to
     * separate the loop level at which storage occurs from the loop
     * level at which computation occurs to trade off between locality
     * and redundant work. This can open the door for two types of
     * optimization.
     *
     * Consider again the pipeline from \ref Func::compute_at :
     \code
     Func f, g;
     Var x, y;
     g(x, y) = x*y;
     f(x, y) = g(x, y) + g(x+1, y) + g(x, y+1) + g(x+1, y+1);
     \endcode
     *
     * If we schedule it like so:
     *
     \code
     g.compute_at(f, x).store_at(f, y);
     \endcode
     *
     * Then the computation of g takes place within the loop over x,
     * but the storage takes place within the loop over y:
     *
     \code
     int f[height][width];
     for (int y = 0; y < height; y++) {
         int g[2][width+1];
         for (int x = 0; x < width; x++) {
             g[0][x] = x*y;
             g[0][x+1] = (x+1)*y;
             g[1][x] = x*(y+1);
             g[1][x+1] = (x+1)*(y+1);
             f[y][x] = g[0][x] + g[1][x] + g[0][x+1] + g[1][x+1];
         }
     }
     \endcode
     *
     * Provided the for loop over x is serial, halide then
     * automatically performs the following sliding window
     * optimization:
     *
     \code
     int f[height][width];
     for (int y = 0; y < height; y++) {
         int g[2][width+1];
         for (int x = 0; x < width; x++) {
             if (x == 0) {
                 g[0][x] = x*y;
                 g[1][x] = x*(y+1);
             }
             g[0][x+1] = (x+1)*y;
             g[1][x+1] = (x+1)*(y+1);
             f[y][x] = g[0][x] + g[1][x] + g[0][x+1] + g[1][x+1];
         }
     }
     \endcode
     *
     * Two of the assignments to g only need to be done when x is
     * zero. The rest of the time, those sites have already been
     * filled in by a previous iteration. This version has the
     * locality of compute_at(f, x), but allocates more memory and
     * does much less redundant work.
     *
     * Halide then further optimizes this pipeline like so:
     *
     \code
     int f[height][width];
     for (int y = 0; y < height; y++) {
         int g[2][2];
         for (int x = 0; x < width; x++) {
             if (x == 0) {
                 g[0][0] = x*y;
                 g[1][0] = x*(y+1);
             }
             g[0][(x+1)%2] = (x+1)*y;
             g[1][(x+1)%2] = (x+1)*(y+1);
             f[y][x] = g[0][x%2] + g[1][x%2] + g[0][(x+1)%2] + g[1][(x+1)%2];
         }
     }
     \endcode
     *
     * Halide has detected that it's possible to use a circular buffer
     * to represent g, and has reduced all accesses to g modulo 2 in
     * the x dimension. This optimization only triggers if the for
     * loop over x is serial, and if halide can statically determine
     * some power of two large enough to cover the range needed. For
     * powers of two, the modulo operator compiles to more efficient
     * bit-masking. This optimization reduces memory usage, and also
     * improves locality by reusing recently-accessed memory instead
     * of pulling new memory into cache.
     *
     */
    EXPORT Func &store_at(Func f, Var var);

    /** Equivalent to the version of store_at that takes a Var, but
     * schedules storage within the loop over a dimension of a
     * reduction domain */
    EXPORT Func &store_at(Func f, RVar var);


    /** Equivalent to the version of store_at that takes a Var, but
     * schedules storage at a given LoopLevel. */
    EXPORT Func &store_at(LoopLevel loop_level);

    /** Equivalent to \ref Func::store_at, but schedules storage
     * outside the outermost loop. */
    EXPORT Func &store_root();

    /** Aggressively inline all uses of this function. This is the
     * default schedule, so you're unlikely to need to call this. For
     * a Func with an update definition, that means it gets computed
     * as close to the innermost loop as possible.
     *
     * Consider once more the pipeline from \ref Func::compute_at :
     *
     \code
     Func f, g;
     Var x, y;
     g(x, y) = x*y;
     f(x, y) = g(x, y) + g(x+1, y) + g(x, y+1) + g(x+1, y+1);
     \endcode
     *
     * Leaving g as inline, this compiles to code equivalent to the following C:
     *
     \code
     int f[height][width];
     for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++) {
             f[y][x] = x*y + x*(y+1) + (x+1)*y + (x+1)*(y+1);
         }
     }
     \endcode
     */
    EXPORT Func &compute_inline();

    /** Get a handle on an update step for the purposes of scheduling
     * it. */
    EXPORT Stage update(int idx = 0);

    /** Trace all loads from this Func by emitting calls to
     * halide_trace. If the Func is inlined, this has no
     * effect. */
    EXPORT Func &trace_loads();

    /** Trace all stores to the buffer backing this Func by emitting
     * calls to halide_trace. If the Func is inlined, this call
     * has no effect. */
    EXPORT Func &trace_stores();

    /** Trace all realizations of this Func by emitting calls to
     * halide_trace. */
    EXPORT Func &trace_realizations();

    /** Get a handle on the internal halide function that this Func
     * represents. Useful if you want to do introspection on Halide
     * functions */
    Internal::Function function() const {
        return func;
    }

    /** You can cast a Func to its pure stage for the purposes of
     * scheduling it. */
    EXPORT operator Stage() const;

    /** Get a handle on the output buffer for this Func. Only relevant
     * if this is the output Func in a pipeline. Useful for making
     * static promises about strides, mins, and extents. */
    // @{
    EXPORT OutputImageParam output_buffer() const;
    EXPORT std::vector<OutputImageParam> output_buffers() const;
    // @}

    /** Use a Func as an argument to an external stage. */
    operator ExternFuncArgument() const {
        return ExternFuncArgument(func);
    }

    /** Infer the arguments to the Func, sorted into a canonical order:
     * all buffers (sorted alphabetically by name), followed by all non-buffers
     * (sorted alphabetically by name).
     This lets you write things like:
     \code
     func.compile_to_assembly("/dev/stdout", func.infer_arguments());
     \endcode
     */
    EXPORT std::vector<Argument> infer_arguments() const;
};

namespace Internal {

template <typename Last>
inline void check_types(const Tuple &t, int idx) {
    using T = typename std::remove_pointer<typename std::remove_reference<Last>::type>::type;
    user_assert(t[idx].type() == type_of<T>())
        << "Can't evaluate expression "
        << t[idx] << " of type " << t[idx].type()
        << " as a scalar of type " << type_of<T>() << "\n";
}

template <typename First, typename Second, typename... Rest>
inline void check_types(const Tuple &t, int idx) {
    check_types<First>(t, idx);
    check_types<Second, Rest...>(t, idx+1);
}

template <typename Last>
inline void assign_results(Realization &r, int idx, Last last) {
    using T = typename std::remove_pointer<typename std::remove_reference<Last>::type>::type;
    *last = Buffer<T>(r[idx])();
}

template <typename First, typename Second, typename... Rest>
inline void assign_results(Realization &r, int idx, First first, Second second, Rest&&... rest) {
    assign_results<First>(r, idx, first);
    assign_results<Second, Rest...>(r, idx+1, second, rest...);
}

}  // namespace Internal

/** JIT-Compile and run enough code to evaluate a Halide
 * expression. This can be thought of as a scalar version of
 * \ref Func::realize */
template<typename T>
NO_INLINE T evaluate(Expr e) {
    user_assert(e.type() == type_of<T>())
        << "Can't evaluate expression "
        << e << " of type " << e.type()
        << " as a scalar of type " << type_of<T>() << "\n";
    Func f;
    f() = e;
    Buffer<T> im = f.realize();
    return im();
}

/** JIT-compile and run enough code to evaluate a Halide Tuple. */
template <typename First, typename... Rest>
NO_INLINE void evaluate(Tuple t, First first, Rest&&... rest) {
    Internal::check_types<First, Rest...>(t, 0);

    Func f;
    f() = t;
    Realization r = f.realize();
    Internal::assign_results(r, 0, first, rest...);
}


namespace Internal {

inline void schedule_scalar(Func f) {
    Target t = get_jit_target_from_environment();
    if (t.has_gpu_feature()) {
        f.gpu_single_thread();
    }
    if (t.has_feature(Target::HVX_64) || t.has_feature(Target::HVX_128)) {
        f.hexagon();
    }
}

}  // namespace Internal

/** JIT-Compile and run enough code to evaluate a Halide
 * expression. This can be thought of as a scalar version of
 * \ref Func::realize. Can use GPU if jit target from environment
 * specifies one.
 */
template<typename T>
NO_INLINE T evaluate_may_gpu(Expr e) {
    user_assert(e.type() == type_of<T>())
        << "Can't evaluate expression "
        << e << " of type " << e.type()
        << " as a scalar of type " << type_of<T>() << "\n";
    Func f;
    f() = e;
    Internal::schedule_scalar(f);
    Buffer<T> im = f.realize();
    return im();
}

/** JIT-compile and run enough code to evaluate a Halide Tuple. Can
 *  use GPU if jit target from environment specifies one. */
// @{
template <typename First, typename... Rest>
NO_INLINE void evaluate_may_gpu(Tuple t, First first, Rest&&... rest) {
    Internal::check_types<First, Rest...>(t, 0);

    Func f;
    f() = t;
    Internal::schedule_scalar(f);
    Realization r = f.realize();
    Internal::assign_results(r, 0, first, rest...);
}
// @}

}


#endif

namespace Halide {

/** namespace to hold functions for imposing boundary conditions on
 *  Halide Funcs.
 *
 *  All functions in this namespace transform a source Func to a
 *  result Func where the result produces the values of the source
 *  within a given region and a different set of values outside the
 *  given region. A region is an N dimensional box specified by
 *  mins and extents.
 *
 *  Three areas are defined:
 *      The image is the entire set of values in the region.
 *      The edge is the set of pixels in the image but adjacent
 *          to coordinates that are not
 *      The interior is the image minus the edge (and is undefined
 *          if the extent of any region is 1 or less).
 *
 *  If the source Func has more dimensions than are specified, the extra ones
 *  are unmodified. Additionally, passing an undefined (default constructed)
 *  'Expr' for the min and extent of a dimension will keep that dimension
 *  unmodified.
 *
 *  Numerous options for specifing the outside area are provided,
 *  including replacement with an expression, repeating the edge
 *  samples, mirroring over the edge, and repeating or mirroring the
 *  entire image.
 *
 *  Using these functions to express your boundary conditions is highly
 *  recommended for correctness and performance. Some of these are hard
 *  to get right. The versions here are both understood by bounds
 *  inference, and also judiciously use the 'likely' intrinsic to minimize
 *  runtime overhead.
 *
 */
namespace BoundaryConditions {

namespace Internal {

inline const Func &func_like_to_func(const Func &func) {
    return func;
}

template <typename T>
inline NO_INLINE Func func_like_to_func(const T &func_like) {
    return lambda(_, func_like(_));
}

}

/** Impose a boundary condition such that a given expression is returned
 *  everywhere outside the boundary. Generally the expression will be a
 *  constant, though the code currently allows accessing the arguments
 *  of source.
 *
 *  An ImageParam, Buffer<T>, or similar can be passed instead of a
 *  Func. If this is done and no bounds are given, the boundaries will
 *  be taken from the min and extent methods of the passed
 *  object. Note that objects are taken by mutable ref. Pipelines
 *  capture Buffers via mutable refs, because running a pipeline might
 *  alter the Buffer metadata (e.g. device allocation state).
 *
 *  (This is similar to setting GL_TEXTURE_WRAP_* to GL_CLAMP_TO_BORDER
 *   and putting value in the border of the texture.)
 *
 *  You may pass undefined Exprs for dimensions that you do not wish
 *  to bound.
 */
// @{
EXPORT Func constant_exterior(const Func &source, Tuple value,
                              const std::vector<std::pair<Expr, Expr>> &bounds);
EXPORT Func constant_exterior(const Func &source, Expr value,
                              const std::vector<std::pair<Expr, Expr>> &bounds);

template <typename T>
inline NO_INLINE Func constant_exterior(const T &func_like, Tuple value) {
    std::vector<std::pair<Expr, Expr>> object_bounds;
    for (int i = 0; i < func_like.dimensions(); i++) {
        object_bounds.push_back({ Expr(func_like.dim(i).min()), Expr(func_like.dim(i).extent()) });
    }

    return constant_exterior(Internal::func_like_to_func(func_like), value, object_bounds);
}
template <typename T>
inline NO_INLINE Func constant_exterior(const T &func_like, Expr value) {
    return constant_exterior(func_like, Tuple(value));
}

template <typename T, typename ...Bounds,
          typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Bounds...>::value>::type* = nullptr>
inline NO_INLINE Func constant_exterior(const T &func_like, Tuple value,
                                        Bounds&&... bounds) {
    std::vector<std::pair<Expr, Expr>> collected_bounds;
    ::Halide::Internal::collect_paired_args(collected_bounds, std::forward<Bounds>(bounds)...);
    return constant_exterior(Internal::func_like_to_func(func_like), value, collected_bounds);
}
template <typename T, typename ...Bounds,
          typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Bounds...>::value>::type* = nullptr>
inline NO_INLINE Func constant_exterior(const T &func_like, Expr value,
                                        Bounds&&... bounds) {
    return constant_exterior(func_like, Tuple(value), std::forward<Bounds>(bounds)...);
}
// @}

/** Impose a boundary condition such that the nearest edge sample is returned
 *  everywhere outside the given region.
 *
 *  An ImageParam, Buffer<T>, or similar can be passed instead of a Func. If this
 *  is done and no bounds are given, the boundaries will be taken from the
 *  min and extent methods of the passed object.
 *
 *  (This is similar to setting GL_TEXTURE_WRAP_* to GL_CLAMP_TO_EDGE.)
 *
 *  You may pass undefined Exprs for dimensions that you do not wish
 *  to bound.
 */
// @{
EXPORT Func repeat_edge(const Func &source,
                        const std::vector<std::pair<Expr, Expr>> &bounds);

template <typename T>
inline NO_INLINE Func repeat_edge(const T &func_like) {
    std::vector<std::pair<Expr, Expr>> object_bounds;
    for (int i = 0; i < func_like.dimensions(); i++) {
        object_bounds.push_back({ Expr(func_like.dim(i).min()), Expr(func_like.dim(i).extent()) });
    }

    return repeat_edge(Internal::func_like_to_func(func_like), object_bounds);
}


template <typename T, typename ...Bounds,
          typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Bounds...>::value>::type* = nullptr>
inline NO_INLINE Func repeat_edge(const T &func_like, Bounds&&... bounds) {
    std::vector<std::pair<Expr, Expr>> collected_bounds;
    ::Halide::Internal::collect_paired_args(collected_bounds, std::forward<Bounds>(bounds)...);
    return repeat_edge(Internal::func_like_to_func(func_like), collected_bounds);
}
// @}

/** Impose a boundary condition such that the entire coordinate space is
 *  tiled with copies of the image abutted against each other.
 *
 *  An ImageParam, Buffer<T>, or similar can be passed instead of a Func. If this
 *  is done and no bounds are given, the boundaries will be taken from the
 *  min and extent methods of the passed object.
 *
 *  (This is similar to setting GL_TEXTURE_WRAP_* to GL_REPEAT.)
 *
 *  You may pass undefined Exprs for dimensions that you do not wish
 *  to bound.
 */
// @{
EXPORT Func repeat_image(const Func &source,
                         const std::vector<std::pair<Expr, Expr>> &bounds);

template <typename T>
inline NO_INLINE Func repeat_image(const T &func_like) {
    std::vector<std::pair<Expr, Expr>> object_bounds;
    for (int i = 0; i < func_like.dimensions(); i++) {
        object_bounds.push_back({ Expr(func_like.dim(i).min()), Expr(func_like.dim(i).extent()) });
    }

    return repeat_image(Internal::func_like_to_func(func_like), object_bounds);
}

template <typename T, typename ...Bounds,
          typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Bounds...>::value>::type* = nullptr>
inline NO_INLINE Func repeat_image(const T &func_like, Bounds&&... bounds) {
    std::vector<std::pair<Expr, Expr>> collected_bounds;
    ::Halide::Internal::collect_paired_args(collected_bounds, std::forward<Bounds>(bounds)...);
    return repeat_image(Internal::func_like_to_func(func_like), collected_bounds);
}

/** Impose a boundary condition such that the entire coordinate space is
 *  tiled with copies of the image abutted against each other, but mirror
 *  them such that adjacent edges are the same.
 *
 *  An ImageParam, Buffer<T>, or similar can be passed instead of a Func. If this
 *  is done and no bounds are given, the boundaries will be taken from the
 *  min and extent methods of the passed object.
 *
 *  (This is similar to setting GL_TEXTURE_WRAP_* to GL_MIRRORED_REPEAT.)
 *
 *  You may pass undefined Exprs for dimensions that you do not wish
 *  to bound.
 */
// @{
EXPORT Func mirror_image(const Func &source,
                         const std::vector<std::pair<Expr, Expr>> &bounds);

template <typename T>
inline NO_INLINE Func mirror_image(const T &func_like) {
    std::vector<std::pair<Expr, Expr>> object_bounds;
    for (int i = 0; i < func_like.dimensions(); i++) {
        object_bounds.push_back({ Expr(func_like.dim(i).min()), Expr(func_like.dim(i).extent()) });
    }

    return mirror_image(Internal::func_like_to_func(func_like), object_bounds);
}

template <typename T, typename ...Bounds,
          typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Bounds...>::value>::type* = nullptr>
inline NO_INLINE Func mirror_image(const T &func_like, Bounds&&... bounds) {
    std::vector<std::pair<Expr, Expr>> collected_bounds;
    ::Halide::Internal::collect_paired_args(collected_bounds, std::forward<Bounds>(bounds)...);
    return mirror_image(Internal::func_like_to_func(func_like), collected_bounds);
}
// @}

/** Impose a boundary condition such that the entire coordinate space is
 *  tiled with copies of the image abutted against each other, but mirror
 *  them such that adjacent edges are the same and then overlap the edges.
 *
 *  This produces an error if any extent is 1 or less. (TODO: check this.)
 *
 *  An ImageParam, Buffer<T>, or similar can be passed instead of a Func. If this
 *  is done and no bounds are given, the boundaries will be taken from the
 *  min and extent methods of the passed object.
 *
 *  (I do not believe there is a direct GL_TEXTURE_WRAP_* equivalent for this.)
 *
 *  You may pass undefined Exprs for dimensions that you do not wish
 *  to bound.
 */
// @{
EXPORT Func mirror_interior(const Func &source,
                            const std::vector<std::pair<Expr, Expr>> &bounds);

template <typename T>
inline NO_INLINE Func mirror_interior(const T &func_like) {
    std::vector<std::pair<Expr, Expr>> object_bounds;
    for (int i = 0; i < func_like.dimensions(); i++) {
        object_bounds.push_back({ Expr(func_like.dim(i).min()), Expr(func_like.dim(i).extent()) });
    }

    return mirror_interior(Internal::func_like_to_func(func_like), object_bounds);
}

template <typename T, typename ...Bounds,
          typename std::enable_if<Halide::Internal::all_are_convertible<Expr, Bounds...>::value>::type* = nullptr>
inline NO_INLINE Func mirror_interior(const T &func_like, Bounds&&... bounds) {
    std::vector<std::pair<Expr, Expr>> collected_bounds;
    ::Halide::Internal::collect_paired_args(collected_bounds, std::forward<Bounds>(bounds)...);
    return mirror_interior(Internal::func_like_to_func(func_like), collected_bounds);
}
// @}

}

}

#endif
#ifndef HALIDE_BOUNDS_INFERENCE_H
#define HALIDE_BOUNDS_INFERENCE_H

/** \file
 * Defines the bounds_inference lowering pass.
 */

#include <map>


namespace Halide {
namespace Internal {

/** Take a partially lowered statement that includes symbolic
 * representations of the bounds over which things should be realized,
 * and inject expressions defining those bounds.
 */
Stmt bounds_inference(Stmt,
                      const std::vector<Function> &outputs,
                      const std::vector<std::string> &realization_order,
                      const std::map<std::string, Function> &environment,
                      const std::map<std::pair<std::string, int>, Interval> &func_bounds,
                      const Target &target);

}
}

#endif
#ifndef HALIDE_CLOSURE_H
#define HALIDE_CLOSURE_H

/** \file
 *
 * Provides Closure class.
 */


namespace Halide {
namespace Internal {

/** A helper class to manage closures. Walks over a statement and
 * retrieves all the references within it to external symbols
 * (variables and allocations). It then helps you build a struct
 * containing the current values of these symbols that you can use as
 * a closure if you want to migrate the body of the statement to its
 * own function (e.g. because it's the body of a parallel for loop. */
class Closure : public IRVisitor {
protected:
    Scope<int> ignore;

    using IRVisitor::visit;

    void visit(const Let *op);
    void visit(const LetStmt *op);
    void visit(const For *op);
    void visit(const Load *op);
    void visit(const Store *op);
    void visit(const Allocate *op);
    void visit(const Variable *op);

public:
    /** Information about a buffer reference from a closure. */
    struct Buffer
    {
        /** The type of the buffer referenced. */
        Type type;

        /** The dimensionality of the buffer. */
        uint8_t dimensions;

        /** The buffer is read from. */
        bool read;

        /** The buffer is written to. */
        bool write;

        /** The size of the buffer if known, otherwise zero. */
        size_t size;

        Buffer() : dimensions(0), read(false), write(false), size(0) { }
    };

protected:
    void found_buffer_ref(const std::string &name, Type type,
                          bool read, bool written, Halide::Buffer<> image);

public:
    Closure() {}

    /** Traverse a statement and find all references to external
     * symbols.
     *
     * When the closure encounters a read or write to 'foo', it
     * assumes that the host pointer is found in the symbol table as
     * 'foo.host', and any buffer_t pointer is found under
     * 'foo.buffer'. */
    Closure(Stmt s, const std::string &loop_variable = "");

    /** External variables referenced. */
    std::map<std::string, Type> vars;

    /** External allocations referenced. */
    std::map<std::string, Buffer> buffers;
};

}}

#endif
#ifndef HALIDE_CODEGEN_ARM_H
#define HALIDE_CODEGEN_ARM_H

/** \file
 * Defines the code-generator for producing ARM machine code
 */

#ifndef HALIDE_CODEGEN_POSIX_H
#define HALIDE_CODEGEN_POSIX_H

/** \file
 * Defines a base-class for code-generators on posixy cpu platforms
 */

#ifndef HALIDE_CODEGEN_LLVM_H
#define HALIDE_CODEGEN_LLVM_H

/** \file
 *
 * Defines the base-class for all architecture-specific code
 * generators that use llvm.
 */

namespace llvm {
class Value;
class Module;
class Function;
class IRBuilderDefaultInserter;
class ConstantFolder;
template<typename, typename> class IRBuilder;
class LLVMContext;
class Type;
class StructType;
class Instruction;
class CallInst;
class ExecutionEngine;
class AllocaInst;
class Constant;
class Triple;
class MDNode;
class NamedMDNode;
class DataLayout;
class BasicBlock;
class GlobalVariable;
}

#include <map>
#include <string>
#include <vector>
#include <memory>


namespace Halide {
namespace Internal {

/** A code generator abstract base class. Actual code generators
 * (e.g. CodeGen_X86) inherit from this. This class is responsible
 * for taking a Halide Stmt and producing llvm bitcode, machine
 * code in an object file, or machine code accessible through a
 * function pointer.
 */
class CodeGen_LLVM : public IRVisitor {
public:
    /** Create an instance of CodeGen_LLVM suitable for the target. */
    static CodeGen_LLVM *new_for_target(const Target &target,
                                        llvm::LLVMContext &context);

    virtual ~CodeGen_LLVM();

    /** Takes a halide Module and compiles it to an llvm Module. */
    virtual std::unique_ptr<llvm::Module> compile(const Module &module);

    /** The target we're generating code for */
    const Target &get_target() const { return target; }

    /** Tell the code generator which LLVM context to use. */
    void set_context(llvm::LLVMContext &context);

    /** Initialize internal llvm state for the enabled targets. */
    static void initialize_llvm();

protected:
    CodeGen_LLVM(Target t);

    /** Compile a specific halide declaration into the llvm Module. */
    // @{
    virtual void compile_func(const LoweredFunc &func, const std::string &simple_name, const std::string &extern_name);
    virtual void compile_buffer(const Buffer<> &buffer);
    // @}

    /** Helper functions for compiling Halide functions to llvm
     * functions. begin_func performs all the work necessary to begin
     * generating code for a function with a given argument list with
     * the IRBuilder. A call to begin_func should be a followed by a
     * call to end_func with the same arguments, to generate the
     * appropriate cleanup code. */
    // @{
    virtual void begin_func(LoweredFunc::LinkageType linkage, const std::string &simple_name,
                            const std::string &extern_name, const std::vector<LoweredArgument> &args);
    virtual void end_func(const std::vector<LoweredArgument> &args);
    // @}

    /** What should be passed as -mcpu, -mattrs, and related for
     * compilation. The architecture-specific code generator should
     * define these. */
    // @{
    virtual std::string mcpu() const = 0;
    virtual std::string mattrs() const = 0;
    virtual bool use_soft_float_abi() const = 0;
    // @}

    /** Should indexing math be promoted to 64-bit on platforms with
     * 64-bit pointers? */
    virtual bool promote_indices() const {return true;}

    /** What's the natural vector bit-width to use for loads, stores, etc. */
    virtual int native_vector_bits() const = 0;

    /** State needed by llvm for code generation, including the
     * current module, function, context, builder, and most recently
     * generated llvm value. */
    //@{
    static bool llvm_initialized;
    static bool llvm_X86_enabled;
    static bool llvm_ARM_enabled;
    static bool llvm_Hexagon_enabled;
    static bool llvm_AArch64_enabled;
    static bool llvm_NVPTX_enabled;
    static bool llvm_Mips_enabled;
    static bool llvm_PowerPC_enabled;

    const Module *input_module;
    std::unique_ptr<llvm::Module> module;
    llvm::Function *function;
    llvm::LLVMContext *context;
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> *builder;
    llvm::Value *value;
    llvm::MDNode *very_likely_branch;
    std::vector<LoweredArgument> current_function_args;
    //@}

    /** The target we're generating code for */
    Halide::Target target;

    /** Grab all the context specific internal state. */
    virtual void init_context();
    /** Initialize the CodeGen_LLVM internal state to compile a fresh
     * module. This allows reuse of one CodeGen_LLVM object to compiled
     * multiple related modules (e.g. multiple device kernels). */
    virtual void init_module();

    /** Add external_code entries to llvm module. */
    void add_external_code(const Module &halide_module);

    /** Run all of llvm's optimization passes on the module. */
    void optimize_module();

    /** Add an entry to the symbol table, hiding previous entries with
     * the same name. Call this when new values come into scope. */
    void sym_push(const std::string &name, llvm::Value *value);

    /** Remove an entry for the symbol table, revealing any previous
     * entries with the same name. Call this when values go out of
     * scope. */
    void sym_pop(const std::string &name);

    /** Fetch an entry from the symbol table. If the symbol is not
     * found, it either errors out (if the second arg is true), or
     * returns nullptr. */
    llvm::Value* sym_get(const std::string &name,
                         bool must_succeed = true) const;

    /** Test if an item exists in the symbol table. */
    bool sym_exists(const std::string &name) const;

    /** Some useful llvm types */
    // @{
    llvm::Type *void_t, *i1_t, *i8_t, *i16_t, *i32_t, *i64_t, *f16_t, *f32_t, *f64_t;
    llvm::StructType *buffer_t_type,
        *type_t_type,
        *dimension_t_type,
        *metadata_t_type,
        *argument_t_type,
        *scalar_value_t_type,
        *device_interface_t_type;
    // @}

    /** Some useful llvm types for subclasses */
    // @{
    llvm::Type *i8x8, *i8x16, *i8x32;
    llvm::Type *i16x4, *i16x8, *i16x16;
    llvm::Type *i32x2, *i32x4, *i32x8;
    llvm::Type *i64x2, *i64x4;
    llvm::Type *f32x2, *f32x4, *f32x8;
    llvm::Type *f64x2, *f64x4;
    // @}

    /** Some wildcard variables used for peephole optimizations in
     * subclasses */
    // @{
    Expr wild_i8x8, wild_i16x4, wild_i32x2; // 64-bit signed ints
    Expr wild_u8x8, wild_u16x4, wild_u32x2; // 64-bit unsigned ints
    Expr wild_i8x16, wild_i16x8, wild_i32x4, wild_i64x2; // 128-bit signed ints
    Expr wild_u8x16, wild_u16x8, wild_u32x4, wild_u64x2; // 128-bit unsigned ints
    Expr wild_i8x32, wild_i16x16, wild_i32x8, wild_i64x4; // 256-bit signed ints
    Expr wild_u8x32, wild_u16x16, wild_u32x8, wild_u64x4; // 256-bit unsigned ints

    Expr wild_f32x2; // 64-bit floats
    Expr wild_f32x4, wild_f64x2; // 128-bit floats
    Expr wild_f32x8, wild_f64x4; // 256-bit floats

    // Wildcards for a varying number of lanes.
    Expr wild_u1x_, wild_i8x_, wild_u8x_, wild_i16x_, wild_u16x_;
    Expr wild_i32x_, wild_u32x_, wild_i64x_, wild_u64x_;
    Expr wild_f32x_, wild_f64x_;
    Expr min_i8, max_i8, max_u8;
    Expr min_i16, max_i16, max_u16;
    Expr min_i32, max_i32, max_u32;
    Expr min_i64, max_i64, max_u64;
    Expr min_f32, max_f32, min_f64, max_f64;
    // @}

    /** Emit code that evaluates an expression, and return the llvm
     * representation of the result of the expression. */
    llvm::Value *codegen(Expr);


    /** Emit code that runs a statement. */
    void codegen(Stmt);

    /** Codegen a vector Expr by codegenning each lane and combining. */
    void scalarize(Expr);

    /** Some destructors should always be called. Others should only
     * be called if the pipeline is exiting with an error code. */
    enum DestructorType {Always, OnError, OnSuccess};

    /* Call this at the location of object creation to register how an
     * object should be destroyed. This does three things:
     * 1) Emits code here that puts the object in a unique
     * null-initialized stack slot
     * 2) Adds an instruction to the destructor block that calls the
     * destructor on that stack slot if it's not null.
     * 3) Returns that stack slot, so you can neuter the destructor
     * (by storing null to the stack slot) or destroy the object early
     * (by calling trigger_destructor).
     */
    llvm::Value *register_destructor(llvm::Function *destructor_fn, llvm::Value *obj, DestructorType when);

    /** Call a destructor early. Pass in the value returned by register destructor. */
    void trigger_destructor(llvm::Function *destructor_fn, llvm::Value *stack_slot);

    /** Retrieves the block containing the error handling
     * code. Creates it if it doesn't already exist for this
     * function. */
    llvm::BasicBlock *get_destructor_block();

    /** Codegen an assertion. If false, returns the error code (if not
     * null), or evaluates and returns the message, which must be an
     * Int(32) expression. */
    // @{
    void create_assertion(llvm::Value *condition, Expr message, llvm::Value *error_code = nullptr);
    // @}

    /** Return the the pipeline with the given error code. Will run
     * the destructor block. */
    void return_with_error_code(llvm::Value *error_code);

    /** Put a string constant in the module as a global variable and return a pointer to it. */
    llvm::Constant *create_string_constant(const std::string &str);

    /** Put a binary blob in the module as a global variable and return a pointer to it. */
    llvm::Constant *create_binary_blob(const std::vector<char> &data, const std::string &name, bool constant = true);

    /** Widen an llvm scalar into an llvm vector with the given number of lanes. */
    llvm::Value *create_broadcast(llvm::Value *, int lanes);

    /** Generate a pointer into a named buffer at a given index, of a
     * given type. The index counts according to the scalar type of
     * the type passed in. */
    // @{
    llvm::Value *codegen_buffer_pointer(std::string buffer, Type type, llvm::Value *index);
    llvm::Value *codegen_buffer_pointer(std::string buffer, Type type, Expr index);
    llvm::Value *codegen_buffer_pointer(llvm::Value *base_address, Type type, Expr index);
    llvm::Value *codegen_buffer_pointer(llvm::Value *base_address, Type type, llvm::Value *index);
    // @}

    /** Turn a Halide Type into an llvm::Value representing a constant halide_type_t */
    llvm::Value *make_halide_type_t(Type);

    /** Mark a load or store with type-based-alias-analysis metadata
     * so that llvm knows it can reorder loads and stores across
     * different buffers */
    void add_tbaa_metadata(llvm::Instruction *inst, std::string buffer, Expr index);

    /** Get a unique name for the actual block of memory that an
     * allocate node uses. Used so that alias analysis understands
     * when multiple Allocate nodes shared the same memory. */
    virtual std::string get_allocation_name(const std::string &n) {return n;}

    /** Helpers for implementing fast integer division. */
    // @{
    // Compute high_half(a*b) >> shr. Note that this is a shift in
    // addition to the implicit shift due to taking the upper half of
    // the multiply result.
    virtual Expr mulhi_shr(Expr a, Expr b, int shr);
    // Compute (a+b)/2, assuming a < b.
    virtual Expr sorted_avg(Expr a, Expr b);
    // @}


    using IRVisitor::visit;

    /** Generate code for various IR nodes. These can be overridden by
     * architecture-specific code to perform peephole
     * optimizations. The result of each is stored in \ref value */
    // @{
    virtual void visit(const IntImm *);
    virtual void visit(const UIntImm *);
    virtual void visit(const FloatImm *);
    virtual void visit(const StringImm *);
    virtual void visit(const Cast *);
    virtual void visit(const Variable *);
    virtual void visit(const Add *);
    virtual void visit(const Sub *);
    virtual void visit(const Mul *);
    virtual void visit(const Div *);
    virtual void visit(const Mod *);
    virtual void visit(const Min *);
    virtual void visit(const Max *);
    virtual void visit(const EQ *);
    virtual void visit(const NE *);
    virtual void visit(const LT *);
    virtual void visit(const LE *);
    virtual void visit(const GT *);
    virtual void visit(const GE *);
    virtual void visit(const And *);
    virtual void visit(const Or *);
    virtual void visit(const Not *);
    virtual void visit(const Select *);
    virtual void visit(const Load *);
    virtual void visit(const Ramp *);
    virtual void visit(const Broadcast *);
    virtual void visit(const Call *);
    virtual void visit(const Let *);
    virtual void visit(const LetStmt *);
    virtual void visit(const AssertStmt *);
    virtual void visit(const ProducerConsumer *);
    virtual void visit(const For *);
    virtual void visit(const Store *);
    virtual void visit(const Block *);
    virtual void visit(const IfThenElse *);
    virtual void visit(const Evaluate *);
    virtual void visit(const Shuffle *);
    virtual void visit(const Prefetch *);
    // @}

    /** Generate code for an allocate node. It has no default
     * implementation - it must be handled in an architecture-specific
     * way. */
    virtual void visit(const Allocate *) = 0;

    /** Generate code for a free node. It has no default
     * implementation and must be handled in an architecture-specific
     * way. */
    virtual void visit(const Free *) = 0;

    /** These IR nodes should have been removed during
     * lowering. CodeGen_LLVM will error out if they are present */
    // @{
    virtual void visit(const Provide *);
    virtual void visit(const Realize *);
    // @}

    /** If we have to bail out of a pipeline midway, this should
     * inject the appropriate target-specific cleanup code. */
    virtual void prepare_for_early_exit() {}

    /** Get the llvm type equivalent to the given halide type in the
     * current context. */
    llvm::Type *llvm_type_of(Type);

    /** Perform an alloca at the function entrypoint. Will be cleaned
     * on function exit. */
    llvm::Value *create_alloca_at_entry(llvm::Type *type, int n,
                                        bool zero_initialize = false,
                                        const std::string &name = "");

    /** Which buffers came in from the outside world (and so we can't
     * guarantee their alignment) */
    std::set<std::string> external_buffer;

    /** The user_context argument. May be a constant null if the
     * function is being compiled without a user context. */
    llvm::Value *get_user_context() const;

    /** Implementation of the intrinsic call to
     * interleave_vectors. This implementation allows for interleaving
     * an arbitrary number of vectors.*/
    virtual llvm::Value *interleave_vectors(const std::vector<llvm::Value *> &);

    /** Generate a call to a vector intrinsic or runtime inlined
     * function. The arguments are sliced up into vectors of the width
     * given by 'intrin_lanes', the intrinsic is called on each
     * piece, then the results (if any) are concatenated back together
     * into the original type 't'. For the version that takes an
     * llvm::Type *, the type may be void, so the vector width of the
     * arguments must be specified explicitly as
     * 'called_lanes'. */
    // @{
    llvm::Value *call_intrin(Type t, int intrin_lanes,
                             const std::string &name, std::vector<Expr>);
    llvm::Value *call_intrin(llvm::Type *t, int intrin_lanes,
                             const std::string &name, std::vector<llvm::Value *>);
    // @}

    /** Take a slice of lanes out of an llvm vector. Pads with undefs
     * if you ask for more lanes than the vector has. */
    virtual llvm::Value *slice_vector(llvm::Value *vec, int start, int extent);

    /** Concatenate a bunch of llvm vectors. Must be of the same type. */
    virtual llvm::Value *concat_vectors(const std::vector<llvm::Value *> &);

    /** Create an LLVM shuffle vectors instruction. */
    virtual llvm::Value *shuffle_vectors(llvm::Value *a, llvm::Value *b,
                                         const std::vector<int> &indices);
    /** Shorthand for shuffling a vector with an undef vector. */
    llvm::Value *shuffle_vectors(llvm::Value *v, const std::vector<int> &indices);

    /** Go looking for a vector version of a runtime function. Will
     * return the best match. Matches in the following order:
     *
     * 1) The requested vector width.
     *
     * 2) The width which is the smallest power of two
     * greater than or equal to the vector width.
     *
     * 3) All the factors of 2) greater than one, in decreasing order.
     *
     * 4) The smallest power of two not yet tried.
     *
     * So for a 5-wide vector, it tries: 5, 8, 4, 2, 16.
     *
     * If there's no match, returns (nullptr, 0).
     */
    std::pair<llvm::Function *, int> find_vector_runtime_function(const std::string &name, int lanes);

    /** Get the result of modulus-remainder analysis for a given expr. */
    ModulusRemainder get_alignment_info(Expr e);

private:

    /** All the values in scope at the current code location during
     * codegen. Use sym_push and sym_pop to access. */
    Scope<llvm::Value *> symbol_table;

    /** Alignment info for Int(32) variables in scope. */
    Scope<ModulusRemainder> alignment_info;

    /** String constants already emitted to the module. Tracked to
     * prevent emitting the same string many times. */
    std::map<std::string, llvm::Constant *> string_constants;

    /** A basic block to branch to on error that triggers all
     * destructors. As destructors are registered, code gets added
     * to this block. */
    llvm::BasicBlock *destructor_block;

    /** Embed an instance of halide_filter_metadata_t in the code, using
     * the given name (by convention, this should be ${FUNCTIONNAME}_metadata)
     * as extern "C" linkage. Note that the return value is a function-returning-
     * pointer-to-constant-data.
     */
    llvm::Function* embed_metadata_getter(const std::string &metadata_getter_name,
        const std::string &function_name, const std::vector<LoweredArgument> &args);

    /** Embed a constant expression as a global variable. */
    llvm::Constant *embed_constant_expr(Expr e);

    llvm::Function *add_argv_wrapper(const std::string &name);

    llvm::Value *codegen_dense_vector_load(const Load *load, llvm::Value *vpred = nullptr);

    virtual void codegen_predicated_vector_load(const Load *op);
    virtual void codegen_predicated_vector_store(const Store *op);
};

}

/** Given a Halide module, generate an llvm::Module. */
EXPORT std::unique_ptr<llvm::Module> codegen_llvm(const Module &module,
                                                  llvm::LLVMContext &context);

}

#endif

namespace Halide {
namespace Internal {

/** A code generator that emits posix code from a given Halide stmt. */
class CodeGen_Posix : public CodeGen_LLVM {
public:

    /** Create an posix code generator. Processor features can be
     * enabled using the appropriate arguments */
    CodeGen_Posix(Target t);

protected:

    using CodeGen_LLVM::visit;

    /** Posix implementation of Allocate. Small constant-sized allocations go
     * on the stack. The rest go on the heap by calling "halide_malloc"
     * and "halide_free" in the standard library. */
    // @{
    void visit(const Allocate *);
    void visit(const Free *);
    // @}

    /** It can be convenient for backends to assume there is extra
     * padding beyond the end of a buffer to enable faster
     * loads/stores. This function gets the padding required by the
     * implementing target. */
    virtual int allocation_padding(Type type) const;

    /** A struct describing heap or stack allocations. */
    struct Allocation {
        /** The memory */
        llvm::Value *ptr;

        /** Destructor stack slot for this allocation. */
        llvm::Value *destructor;

        /** Function to accomplish the destruction. */
        llvm::Function *destructor_function;

        /** The (Halide) type of the allocation. */
        Type type;

        /** How many bytes this allocation is, or 0 if not
         * constant. */
        int constant_bytes;

        /** How many bytes of stack space used. 0 implies it was a
         * heap allocation. */
        int stack_bytes;

        /** A unique name for this allocation. May not be equal to the
         * Allocate node name in cases where we detect multiple
         * Allocate nodes can share a single allocation. */
        std::string name;
    };

    /** The allocations currently in scope. The stack gets pushed when
     * we enter a new function. */
    Scope<Allocation> allocations;

    std::string get_allocation_name(const std::string &n);

private:

    /** Stack allocations that were freed, but haven't gone out of
     * scope yet.  This allows us to re-use stack allocations when
     * they aren't being used. */
    std::vector<Allocation> free_stack_allocs;

    /** current size of all alloca instances in use; this is tracked only
     * for debug output purposes. */
    size_t cur_stack_alloc_total{0};

    /** Generates code for computing the size of an allocation from a
     * list of its extents and its size. Fires a runtime assert
     * (halide_error) if the size overflows 2^31 -1, the maximum
     * positive number an int32_t can hold. */
    llvm::Value *codegen_allocation_size(const std::string &name, Type type, const std::vector<Expr> &extents);

    /** Allocates some memory on either the stack or the heap, and
     * returns an Allocation object describing it. For heap
     * allocations this calls halide_malloc in the runtime, and for
     * stack allocations it either reuses an existing block from the
     * free_stack_blocks list, or it saves the stack pointer and calls
     * alloca.
     *
     * This call returns the allocation, pushes it onto the
     * 'allocations' map, and adds an entry to the symbol table called
     * name.host that provides the base pointer.
     *
     * When the allocation can be freed call 'free_allocation', and
     * when it goes out of scope call 'destroy_allocation'. */
    Allocation create_allocation(const std::string &name, Type type,
                                 const std::vector<Expr> &extents,
                                 Expr condition, Expr new_expr, std::string free_function);

    /** Free an allocation previously allocated with
     * create_allocation */
    void free_allocation(const std::string &name);

};

}}

#endif

namespace Halide {
namespace Internal {

/** A code generator that emits ARM code from a given Halide stmt. */
class CodeGen_ARM : public CodeGen_Posix {
public:
    /** Create an ARM code generator for the given arm target. */
    CodeGen_ARM(Target);

protected:

    Expr sorted_avg(Expr a, Expr b);

    using CodeGen_Posix::visit;

    /** Nodes for which we want to emit specific neon intrinsics */
    // @{
    void visit(const Cast *);
    void visit(const Add *);
    void visit(const Sub *);
    void visit(const Div *);
    void visit(const Mul *);
    void visit(const Min *);
    void visit(const Max *);
    void visit(const Store *);
    void visit(const Load *);
    void visit(const Call *);
    // @}

    /** Various patterns to peephole match against */
    struct Pattern {
        std::string intrin32; ///< Name of the intrinsic for 32-bit arm
        std::string intrin64; ///< Name of the intrinsic for 64-bit arm
        int intrin_lanes;     ///< The native vector width of the intrinsic
        Expr pattern;         ///< The pattern to match against
        enum PatternType {Simple = 0, ///< Just match the pattern
                          LeftShift,  ///< Match the pattern if the RHS is a const power of two
                          RightShift, ///< Match the pattern if the RHS is a const power of two
                          NarrowArgs  ///< Match the pattern if the args can be losslessly narrowed
        };
        PatternType type;
        Pattern() {}
        Pattern(const std::string &i32, const std::string &i64, int l, Expr p, PatternType t = Simple) :
            intrin32("llvm.arm.neon." + i32),
            intrin64("llvm.aarch64.neon." + i64),
            intrin_lanes(l), pattern(p), type(t) {}
    };
    std::vector<Pattern> casts, left_shifts, averagings, negations;

    // Call an intrinsic as defined by a pattern. Dispatches to the
    // 32- or 64-bit name depending on the target's bit width.
    // @{
    llvm::Value *call_pattern(const Pattern &p, Type t, const std::vector<Expr> &args);
    llvm::Value *call_pattern(const Pattern &p, llvm::Type *t, const std::vector<llvm::Value *> &args);
    // @}

    std::string mcpu() const;
    std::string mattrs() const;
    bool use_soft_float_abi() const;
    int native_vector_bits() const;

    // NEON can be disabled for older processors.
    bool neon_intrinsics_disabled() {
        return target.has_feature(Target::NoNEON);
    }
};

}}

#endif
#ifndef HALIDE_CODEGEN_C_H
#define HALIDE_CODEGEN_C_H

/** \file
 *
 * Defines an IRPrinter that emits C++ code equivalent to a halide stmt
 */

#ifndef HALIDE_IR_PRINTER_H
#define HALIDE_IR_PRINTER_H

/** \file
 * This header file defines operators that let you dump a Halide
 * expression, statement, or type directly into an output stream
 * in a human readable form.
 * E.g:
 \code
 Expr foo = ...
 std::cout << "Foo is " << foo << std::endl;
 \endcode
 *
 * These operators are implemented using \ref Halide::Internal::IRPrinter
 */

#include <ostream>


namespace Halide {

/** Emit an expression on an output stream (such as std::cout) in a
 * human-readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const Expr &);

/** Emit a halide type on an output stream (such as std::cout) in a
 * human-readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const Type &);

/** Emit a halide Module on an output stream (such as std::cout) in a
 * human-readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const Module &);

/** Emit a halide device api type in a human readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const DeviceAPI &);

/** Emit a halide LoopLevel in a human readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const LoopLevel &);

namespace Internal {

struct AssociativePattern;
struct AssociativeOp;

/** Emit a halide associative pattern on an output stream (such as std::cout)
 * in a human-readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const AssociativePattern &);

/** Emit a halide associative op on an output stream (such as std::cout) in a
 * human-readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const AssociativeOp &);

/** Emit a halide statement on an output stream (such as std::cout) in
 * a human-readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const Stmt &);

/** Emit a halide for loop type (vectorized, serial, etc) in a human
 * readable form */
EXPORT std::ostream &operator<<(std::ostream &stream, const ForType &);

/** Emit a halide name mangling value in a human readable format */
EXPORT std::ostream &operator<<(std::ostream &stream, const NameMangling &);

/** An IRVisitor that emits IR to the given output stream in a human
 * readable form. Can be subclassed if you want to modify the way in
 * which it prints.
 */
class IRPrinter : public IRVisitor {
public:
    EXPORT virtual ~IRPrinter();

    /** Construct an IRPrinter pointed at a given output stream
     * (e.g. std::cout, or a std::ofstream) */
    EXPORT IRPrinter(std::ostream &);

    /** emit an expression on the output stream */
    EXPORT void print(Expr);

    /** emit a statement on the output stream */
    EXPORT void print(Stmt);

    /** emit a comma delimited list of exprs, without any leading or
     * trailing punctuation. */
    EXPORT void print_list(const std::vector<Expr> &exprs);

    EXPORT static void test();

protected:
    /** The stream we're outputting on */
    std::ostream &stream;

    /** The current indentation level, useful for pretty-printing
     * statements */
    int indent;

    /** Emit spaces according to the current indentation level */
    void do_indent();

    void visit(const IntImm *);
    void visit(const UIntImm *);
    void visit(const FloatImm *);
    void visit(const StringImm *);
    void visit(const Cast *);
    void visit(const Variable *);
    void visit(const Add *);
    void visit(const Sub *);
    void visit(const Mul *);
    void visit(const Div *);
    void visit(const Mod *);
    void visit(const Min *);
    void visit(const Max *);
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GT *);
    void visit(const GE *);
    void visit(const And *);
    void visit(const Or *);
    void visit(const Not *);
    void visit(const Select *);
    void visit(const Load *);
    void visit(const Ramp *);
    void visit(const Broadcast *);
    void visit(const Call *);
    void visit(const Let *);
    void visit(const LetStmt *);
    void visit(const AssertStmt *);
    void visit(const ProducerConsumer *);
    void visit(const For *);
    void visit(const Store *);
    void visit(const Provide *);
    void visit(const Allocate *);
    void visit(const Free *);
    void visit(const Realize *);
    void visit(const Block *);
    void visit(const IfThenElse *);
    void visit(const Evaluate *);
    void visit(const Shuffle *);
    void visit(const Prefetch *);
};
}
}

#endif

namespace Halide {

struct Argument;

namespace Internal {

/** This class emits C++ code equivalent to a halide Stmt. It's
 * mostly the same as an IRPrinter, but it's wrapped in a function
 * definition, and some things are handled differently to be valid
 * C++.
 */
class CodeGen_C : public IRPrinter {
public:
    enum OutputKind {
        CHeader,
        CPlusPlusHeader,
        CImplementation,
        CPlusPlusImplementation,
    };

    /** Initialize a C code generator pointing at a particular output
     * stream (e.g. a file, or std::cout) */
    CodeGen_C(std::ostream &dest,
              Target target,
              OutputKind output_kind = CImplementation,
              const std::string &include_guard = "");
    ~CodeGen_C();

    /** Emit the declarations contained in the module as C code. */
    void compile(const Module &module);

    /** The target we're generating code for */
    const Target &get_target() const { return target; }

    EXPORT static void test();

protected:

    /** Emit a declaration. */
    // @{
    virtual void compile(const LoweredFunc &func);
    virtual void compile(const Buffer<> &buffer);
    // @}

    /** An ID for the most recently generated ssa variable */
    std::string id;

    /** The target being generated for. */
    Target target;

    /** Controls whether this instance is generating declarations or
     * definitions and whether the interface us extern "C" or C++. */
    OutputKind output_kind;

    /** A cache of generated values in scope */
    std::map<std::string, std::string> cache;

    /** Emit an expression as an assignment, then return the id of the
     * resulting var */
    std::string print_expr(Expr);

    /** Like print_expr, but cast the Expr to the given Type */
    std::string print_cast_expr(const Type &, Expr);

    /** Emit a statement */
    void print_stmt(Stmt);

    void create_assertion(const std::string &id_cond, const std::string &id_msg);
    void create_assertion(const std::string &id_cond, Expr message);
    void create_assertion(Expr cond, Expr message);

    enum AppendSpaceIfNeeded {
        DoNotAppendSpace,
        AppendSpace,
    };

    /** Emit the C name for a halide type. If space_option is AppendSpace,
     *  and there should be a space between the type and the next token,
     *  one is appended. (This allows both "int foo" and "Foo *foo" to be
     *  formatted correctly. Otherwise the latter is "Foo * foo".)
     */
    virtual std::string print_type(Type, AppendSpaceIfNeeded space_option = DoNotAppendSpace);

    /** Emit a statement to reinterpret an expression as another type */
    virtual std::string print_reinterpret(Type, Expr);

    /** Emit a version of a string that is a valid identifier in C (. is replaced with _) */
    virtual std::string print_name(const std::string &);

    /** Add typedefs for vector types. Not needed for OpenCL, might
     * use different syntax for other C-like languages. */
    virtual void add_vector_typedefs(const std::set<Type> &vector_types);

    /** Bottleneck to allow customization of calls to generic Extern/PureExtern calls.  */
    virtual std::string print_extern_call(const Call *op);

    /** Convert a vector Expr into a series of scalar Exprs, then reassemble into vector of original type.  */
    std::string print_scalarized_expr(Expr e);

    /** Emit an SSA-style assignment, and set id to the freshly generated name. Return id. */
    std::string print_assignment(Type t, const std::string &rhs);

    /** Return true if only generating an interface, which may be extern "C" or C++ */
    bool is_header() {
        return output_kind == CHeader ||
               output_kind == CPlusPlusHeader;
    }

    /** Return true if generating C++ linkage. */
    bool is_c_plus_plus_interface() {
        return output_kind == CPlusPlusHeader ||
               output_kind == CPlusPlusImplementation;
    }

    /** Open a new C scope (i.e. throw in a brace, increase the indent) */
    void open_scope();

    /** Close a C scope (i.e. throw in an end brace, decrease the indent) */
    void close_scope(const std::string &comment);

    struct Allocation {
        Type type;
    };

    /** Track the types of allocations to avoid unnecessary casts. */
    Scope<Allocation> allocations;

    /** Track which allocations actually went on the heap. */
    Scope<int> heap_allocations;

    /** True if there is a void * __user_context parameter in the arguments. */
    bool have_user_context;

    /** Track current calling convention scope. */
    bool extern_c_open;

    /** True if at least one gpu-based for loop is used. */
    bool uses_gpu_for_loops;

    /** Track which handle types have been forward-declared already. */
    std::set<const halide_handle_cplusplus_type *> forward_declared;

    /** If the Type is a handle type, emit a forward-declaration for it
     * if we haven't already. */
    void forward_declare_type_if_needed(const Type &t);

    void set_name_mangling_mode(NameMangling mode);

    using IRPrinter::visit;

    void visit(const Variable *);
    void visit(const IntImm *);
    void visit(const UIntImm *);
    void visit(const StringImm *);
    void visit(const FloatImm *);
    void visit(const Cast *);
    void visit(const Add *);
    void visit(const Sub *);
    void visit(const Mul *);
    void visit(const Div *);
    void visit(const Mod *);
    void visit(const Max *);
    void visit(const Min *);
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GT *);
    void visit(const GE *);
    void visit(const And *);
    void visit(const Or *);
    void visit(const Not *);
    void visit(const Call *);
    void visit(const Select *);
    void visit(const Load *);
    void visit(const Store *);
    void visit(const Let *);
    void visit(const LetStmt *);
    void visit(const AssertStmt *);
    void visit(const ProducerConsumer *);
    void visit(const For *);
    void visit(const Ramp *);
    void visit(const Broadcast *);
    void visit(const Provide *);
    void visit(const Allocate *);
    void visit(const Free *);
    void visit(const Realize *);
    void visit(const IfThenElse *);
    void visit(const Evaluate *);
    void visit(const Shuffle *);
    void visit(const Prefetch *);

    void visit_binop(Type t, Expr a, Expr b, const char *op);

    template<typename T>
    static std::string with_sep(const std::vector<T> &v, const std::string &sep) {
        std::ostringstream o;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
                o << sep;
            }
            o << v[i];
        }
        return o.str();
    }

    template<typename T>
    static std::string with_commas(const std::vector<T> &v) {
        return with_sep<T>(v, ", ");
    }
};

}
}

#endif
#ifndef HALIDE_CODEGEN_GPU_DEV_H
#define HALIDE_CODEGEN_GPU_DEV_H

/** \file
 * Defines the code-generator interface for producing GPU device code
 */

#ifndef HALIDE_DEVICE_ARGUMENT_H
#define HALIDE_DEVICE_ARGUMENT_H

/** \file
 * Defines helpers for passing arguments to separate devices, such as GPUs.
 */


namespace Halide {
namespace Internal {

/** A DeviceArgument looks similar to an Halide::Argument, but has behavioral
 * differences that make it specific to the GPU pipeline; the fact that
 * neither is-a nor has-a Halide::Argument is deliberate. In particular, note
 * that a Halide::Argument that is a buffer can be read or write, but not both,
 * while a DeviceArgument that is a buffer can be read *and* write for some GPU
 * backends. */
struct DeviceArgument {
    /** The name of the argument */
    std::string name;

    /** An argument is either a primitive type (for parameters), or a
     * buffer pointer.
     *
     * If is_buffer == false, then type fully encodes the expected type
     * of the scalar argument.
     *
     * If is_buffer == true, then type.bytes() should be used to determine
     * elem_size of the buffer; additionally, type.code *should* reflect
     * the expected interpretation of the buffer data (e.g. float vs int),
     * but there is no runtime enforcement of this at present.
     */
    bool is_buffer;

    /** If is_buffer is true, this is the dimensionality of the buffer.
     * If is_buffer is false, this value is ignored (and should always be set to zero) */
    uint8_t dimensions;

    /** If this is a scalar parameter, then this is its type.
     *
     * If this is a buffer parameter, this is used to determine elem_size
     * of the buffer_t.
     *
     * Note that type.lanes() should always be 1 here. */
    Type type;

    /** The static size of the argument if known, or zero otherwise. */
    size_t size;

    /** The index of the first element of the argument when packed into a wider
     * type, such as packing scalar floats into vec4 for GLSL. */
    size_t packed_index;

    /** For buffers, these two variables can be used to specify whether the
     * buffer is read or written. By default, we assume that the argument
     * buffer is read-write and set both flags. */
    bool read;
    bool write;

    DeviceArgument() :
        is_buffer(false),
        dimensions(0),
        size(0),
        packed_index(0),
        read(false),
        write(false) {}

    DeviceArgument(const std::string &_name,
                   bool _is_buffer,
                   Type _type,
                   uint8_t _dimensions,
                   size_t _size = 0) :
        name(_name),
        is_buffer(_is_buffer),
        dimensions(_dimensions),
        type(_type),
        size(_size),
        packed_index(0),
        read(_is_buffer),
        write(_is_buffer) {}
};

/** A Closure modified to inspect GPU-specific memory accesses, and
 * produce a vector of DeviceArgument objects. */
class HostClosure : public Closure {
public:
    HostClosure(Stmt s, const std::string &loop_variable = "");

    /** Get a description of the captured arguments. */
    std::vector<DeviceArgument> arguments();

protected:
    using Internal::Closure::visit;
    void visit(const For *loop);
    void visit(const Call *op);
};

}}

#endif

namespace Halide {
namespace Internal {

/** A code generator that emits GPU code from a given Halide stmt. */
struct CodeGen_GPU_Dev {
    virtual ~CodeGen_GPU_Dev();

    /** Compile a GPU kernel into the module. This may be called many times
     * with different kernels, which will all be accumulated into a single
     * source module shared by a given Halide pipeline. */
    virtual void add_kernel(Stmt stmt,
                            const std::string &name,
                            const std::vector<DeviceArgument> &args) = 0;

    /** (Re)initialize the GPU kernel module. This is separate from compile,
     * since a GPU device module will often have many kernels compiled into it
     * for a single pipeline. */
    virtual void init_module() = 0;

    virtual std::vector<char> compile_to_src() = 0;

    virtual std::string get_current_kernel_name() = 0;

    virtual void dump() = 0;

    /** This routine returns the GPU API name that is combined into
     *  runtime routine names to ensure each GPU API has a unique
     *  name.
     */
    virtual std::string api_unique_name() = 0;

    /** Returns the specified name transformed by the variable naming rules
     * for the GPU language backend. Used to determine the name of a parameter
     * during host codegen. */
    virtual std::string print_gpu_name(const std::string &name) = 0;

    static bool is_gpu_var(const std::string &name);
    static bool is_gpu_block_var(const std::string &name);
    static bool is_gpu_thread_var(const std::string &name);

    /** Checks if expr is block uniform, i.e. does not depend on a thread
     * var. */
    static bool is_block_uniform(Expr expr);
    /** Checks if the buffer is a candidate for constant storage. Most
     * GPUs (APIs) support a constant memory storage class that cannot be
     * written to and performs well for block uniform accesses. A buffer is a
     * candidate for constant storage if it is never written to, and loads are
     * uniform within the workgroup. */
    static bool is_buffer_constant(Stmt kernel, const std::string &buffer);

    /** Return the total size of an allocation. If the size is not constant,
     * this returns its upper bound. If the result overflows, this throws an
     * assertion. If there is no constant upper bound, this returns 0. */
    static int32_t get_constant_bound_allocation_size(const Allocate *alloc);
};

}}

#endif
#ifndef HALIDE_CODEGEN_GPU_HOST_H
#define HALIDE_CODEGEN_GPU_HOST_H

/** \file
 * Defines the code-generator for producing GPU host code
 */

#include <map>

#ifndef HALIDE_CODEGEN_X86_H
#define HALIDE_CODEGEN_X86_H

/** \file
 * Defines the code-generator for producing x86 machine code
 */


namespace llvm {
class JITEventListener;
}

namespace Halide {
namespace Internal {

/** A code generator that emits x86 code from a given Halide stmt. */
class CodeGen_X86 : public CodeGen_Posix {
public:
    /** Create an x86 code generator. Processor features can be
     * enabled using the appropriate flags in the target struct. */
    CodeGen_X86(Target);

protected:

    std::string mcpu() const;
    std::string mattrs() const;
    bool use_soft_float_abi() const;
    int native_vector_bits() const;

    Expr mulhi_shr(Expr a, Expr b, int shr);

    using CodeGen_Posix::visit;

    /** Nodes for which we want to emit specific sse/avx intrinsics */
    // @{
    void visit(const Call *);
    void visit(const Add *);
    void visit(const Sub *);
    void visit(const Cast *);
    void visit(const GT *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GE *);
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const Select *);
    // @}
};

}}

#endif
#ifndef HALIDE_CODEGEN_MIPS_H
#define HALIDE_CODEGEN_MIPS_H

/** \file
 * Defines the code-generator for producing MIPS machine code.
 */


namespace Halide {
namespace Internal {

/** A code generator that emits mips code from a given Halide stmt. */
class CodeGen_MIPS : public CodeGen_Posix {
public:
    /** Create a mips code generator. Processor features can be
     * enabled using the appropriate flags in the target struct. */
    CodeGen_MIPS(Target);

    static void test();

protected:

    using CodeGen_Posix::visit;

    std::string mcpu() const;
    std::string mattrs() const;
    bool use_soft_float_abi() const;
    int native_vector_bits() const;
};

}}

#endif
#ifndef HALIDE_CODEGEN_POWERPC_H
#define HALIDE_CODEGEN_POWERPC_H

/** \file
 * Defines the code-generator for producing POWERPC machine code.
 */


namespace Halide {
namespace Internal {

/** A code generator that emits mips code from a given Halide stmt. */
class CodeGen_PowerPC : public CodeGen_Posix {
public:
    /** Create a powerpc code generator. Processor features can be
     * enabled using the appropriate flags in the target struct. */
    CodeGen_PowerPC(Target);

    static void test();

protected:

    std::string mcpu() const;
    std::string mattrs() const;
    bool use_soft_float_abi() const;
    int native_vector_bits() const;

    using CodeGen_Posix::visit;

    /** Nodes for which we want to emit specific sse/avx intrinsics */
    // @{
    void visit(const Cast *);
    void visit(const Min *);
    void visit(const Max *);
    // @}

    // Call an intrinsic as defined by a pattern. Dispatches to the
private:
    static const char* altivec_int_type_name(const Type&);
};

}}

#endif


namespace Halide {
namespace Internal {

struct CodeGen_GPU_Dev;
struct GPU_Argument;

/** A code generator that emits GPU code from a given Halide stmt. */
template<typename CodeGen_CPU>
class CodeGen_GPU_Host : public CodeGen_CPU {
public:

    /** Create a GPU code generator. GPU target is selected via
     * CodeGen_GPU_Options. Processor features can be enabled using the
     * appropriate flags from Target */
    CodeGen_GPU_Host(Target);

    virtual ~CodeGen_GPU_Host();

protected:
    void compile_func(const LoweredFunc &func, const std::string &simple_name, const std::string &extern_name);

    /** Declare members of the base class that must exist to help the
     * compiler do name lookup. Annoying but necessary, because the
     * compiler doesn't know that CodeGen_CPU will in fact inherit
     * from CodeGen for every instantiation of this template. */
    using CodeGen_CPU::module;
    using CodeGen_CPU::init_module;
    using CodeGen_CPU::target;
    using CodeGen_CPU::builder;
    using CodeGen_CPU::context;
    using CodeGen_CPU::function;
    using CodeGen_CPU::get_user_context;
    using CodeGen_CPU::visit;
    using CodeGen_CPU::codegen;
    using CodeGen_CPU::sym_push;
    using CodeGen_CPU::sym_pop;
    using CodeGen_CPU::sym_get;
    using CodeGen_CPU::sym_exists;
    using CodeGen_CPU::llvm_type_of;
    using CodeGen_CPU::create_alloca_at_entry;
    using CodeGen_CPU::i8_t;
    using CodeGen_CPU::i32_t;
    using CodeGen_CPU::i64_t;
    using CodeGen_CPU::buffer_t_type;
    using CodeGen_CPU::allocations;
    using CodeGen_CPU::register_destructor;

    /** Nodes for which we need to override default behavior for the GPU runtime */
    // @{
    void visit(const For *);
    // @}

    std::string function_name;

    llvm::Value *get_module_state(const std::string &api_unique_name,
                                  bool create = true);

private:
    /** Child code generator for device kernels. */
    std::map<DeviceAPI, CodeGen_GPU_Dev *> cgdev;
};

}}

#endif
#ifndef HALIDE_CODEGEN_OPENCL_DEV_H
#define HALIDE_CODEGEN_OPENCL_DEV_H

/** \file
 * Defines the code-generator for producing OpenCL C kernel code
 */

#include <sstream>


namespace Halide {
namespace Internal {

class CodeGen_OpenCL_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_OpenCL_Dev(Target target);

    /** Compile a GPU kernel into the module. This may be called many times
     * with different kernels, which will all be accumulated into a single
     * source module shared by a given Halide pipeline. */
    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args);

    /** (Re)initialize the GPU kernel module. This is separate from compile,
     * since a GPU device module will often have many kernels compiled into it
     * for a single pipeline. */
    void init_module();

    std::vector<char> compile_to_src();

    std::string get_current_kernel_name();

    void dump();

    virtual std::string print_gpu_name(const std::string &name);

    std::string api_unique_name() { return "opencl"; }

protected:

    class CodeGen_OpenCL_C : public CodeGen_C {
    public:
        CodeGen_OpenCL_C(std::ostream &s, Target t) : CodeGen_C(s, t) {}
        void add_kernel(Stmt stmt,
                        const std::string &name,
                        const std::vector<DeviceArgument> &args);

    protected:
        using CodeGen_C::visit;
        std::string print_type(Type type, AppendSpaceIfNeeded append_space = DoNotAppendSpace);
        std::string print_reinterpret(Type type, Expr e);
        std::string print_extern_call(const Call *op);
        void add_vector_typedefs(const std::set<Type> &vector_types);

        std::string get_memory_space(const std::string &);

        void visit(const For *);
        void visit(const Ramp *op);
        void visit(const Broadcast *op);
        void visit(const Call *op);
        void visit(const Load *op);
        void visit(const Store *op);
        void visit(const Cast *op);
        void visit(const Select *op);
        void visit(const EQ *);
        void visit(const NE *);
        void visit(const LT *);
        void visit(const LE *);
        void visit(const GT *);
        void visit(const GE *);
        void visit(const Allocate *op);
        void visit(const Free *op);
        void visit(const AssertStmt *op);
        void visit(const Shuffle *op);
        void visit(const Min *op);
        void visit(const Max *op);
    };

    std::ostringstream src_stream;
    std::string cur_kernel_name;
    CodeGen_OpenCL_C clc;
};

}}

#endif
#ifndef HALIDE_CODEGEN_METAL_DEV_H
#define HALIDE_CODEGEN_METAL_DEV_H

/** \file
 * Defines the code-generator for producing Apple Metal shading language kernel code
 */

#include <sstream>


namespace Halide {
namespace Internal {

class CodeGen_Metal_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_Metal_Dev(Target target);

    /** Compile a GPU kernel into the module. This may be called many times
     * with different kernels, which will all be accumulated into a single
     * source module shared by a given Halide pipeline. */
    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args);

    /** (Re)initialize the GPU kernel module. This is separate from compile,
     * since a GPU device module will often have many kernels compiled into it
     * for a single pipeline. */
    void init_module();

    std::vector<char> compile_to_src();

    std::string get_current_kernel_name();

    void dump();

    virtual std::string print_gpu_name(const std::string &name);

    std::string api_unique_name() { return "metal"; }

protected:

    class CodeGen_Metal_C : public CodeGen_C {
    public:
        CodeGen_Metal_C(std::ostream &s, Target t) : CodeGen_C(s, t) {}
        void add_kernel(Stmt stmt,
                        const std::string &name,
                        const std::vector<DeviceArgument> &args);

    protected:
        using CodeGen_C::visit;
        std::string print_type(Type type, AppendSpaceIfNeeded space_option = DoNotAppendSpace);
        // Vectors in Metal come in two varieties, regular and packed.
        // For storage allocations and pointers used in address arithmetic,
        // packed types must be used. For temporaries, constructors, etc.
        // regular types must be used.
        // This concept also potentially applies to half types, which are
        // often only supported for storage, not arithmetic,
        // hence the method name.
        std::string print_storage_type(Type type);
        std::string print_type_maybe_storage(Type type, bool storage, AppendSpaceIfNeeded space);
        std::string print_reinterpret(Type type, Expr e);
        std::string print_extern_call(const Call *op);

        std::string get_memory_space(const std::string &);

        void visit(const Min *);
        void visit(const Max *);
        void visit(const Div *);
        void visit(const Mod *);
        void visit(const For *);
        void visit(const Ramp *op);
        void visit(const Broadcast *op);
        void visit(const Load *op);
        void visit(const Store *op);
        void visit(const Select *op);
        void visit(const Allocate *op);
        void visit(const Free *op);
        void visit(const Cast *op);
    };

    std::ostringstream src_stream;
    std::string cur_kernel_name;
    CodeGen_Metal_C metal_c;
};

}}

#endif
#ifndef HALIDE_CODEGEN_OPENGL_DEV_H
#define HALIDE_CODEGEN_OPENGL_DEV_H

/** \file
 * Defines the code-generator for producing GLSL kernel code
 */

#include <sstream>
#include <map>


namespace Halide {
namespace Internal {

class CodeGen_GLSL;

class CodeGen_OpenGL_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_OpenGL_Dev(const Target &target);
    ~CodeGen_OpenGL_Dev();

    // CodeGen_GPU_Dev interface
    void add_kernel(Stmt stmt, const std::string &name,
                    const std::vector<DeviceArgument> &args);

    void init_module();

    std::vector<char> compile_to_src();

    std::string get_current_kernel_name();

    void dump();

    std::string api_unique_name() { return "opengl"; }

private:
    CodeGen_GLSL *glc;

    virtual std::string print_gpu_name(const std::string &name);

private:
    std::ostringstream src_stream;
    std::string cur_kernel_name;
    Target target;
};

/**
  * This class handles GLSL arithmetic, shared by CodeGen_GLSL and CodeGen_OpenGLCompute_C.
  */
class CodeGen_GLSLBase : public CodeGen_C {
public:
    CodeGen_GLSLBase(std::ostream &s, Target t);

    std::string print_name(const std::string &name);
    std::string print_type(Type type, AppendSpaceIfNeeded space_option = DoNotAppendSpace);

protected:
    using CodeGen_C::visit;
    void visit(const Max *op);
    void visit(const Min *op);
    void visit(const Div *op);
    void visit(const Mod *op);
    void visit(const Call *op);

    // these have specific functions
    // in GLSL that operate on vectors
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GT *);
    void visit(const GE *);

    void visit(const Shuffle *);

private:
    std::map<std::string, std::string> builtin;
};


/** Compile one statement into GLSL. */
class CodeGen_GLSL : public CodeGen_GLSLBase {
public:
    CodeGen_GLSL(std::ostream &s, const Target &t) : CodeGen_GLSLBase(s, t) {}

    void add_kernel(Stmt stmt,
                    std::string name,
                    const std::vector<DeviceArgument> &args);

    EXPORT static void test();

protected:
    using CodeGen_C::visit;

    void visit(const FloatImm *);
    void visit(const UIntImm *);
    void visit(const IntImm *);

    void visit(const Cast *);
    void visit(const Let *);
    void visit(const For *);
    void visit(const Select *);

    void visit(const Load *);
    void visit(const Store *);
    void visit(const Allocate *);
    void visit(const Free *);

    void visit(const Call *);
    void visit(const AssertStmt *);
    void visit(const Ramp *op);
    void visit(const Broadcast *);

    void visit(const Evaluate *);

private:
    std::string get_vector_suffix(Expr e);

    std::vector<std::string> print_lanes(Expr expr);

    Scope<int> scalar_vars, vector_vars;
};

}}

#endif
#ifndef HALIDE_CODEGEN_OPENGLCOMPUTE_DEV_H
#define HALIDE_CODEGEN_OPENGLCOMPUTE_DEV_H

/** \file
 * Defines the code-generator for producing GLSL kernel code for OpenGL Compute.
 */

#include <sstream>
#include <map>


namespace Halide {
namespace Internal {

class CodeGen_OpenGLCompute_Dev : public CodeGen_GPU_Dev {
public:
    CodeGen_OpenGLCompute_Dev(Target target);

    // CodeGen_GPU_Dev interface
    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args);

    void init_module();

    std::vector<char> compile_to_src();

    std::string get_current_kernel_name();

    void dump();

    virtual std::string print_gpu_name(const std::string &name);

    std::string api_unique_name() { return "openglcompute"; }

protected:

    class CodeGen_OpenGLCompute_C : public CodeGen_GLSLBase {
    public:
        CodeGen_OpenGLCompute_C(std::ostream &s, Target t) : CodeGen_GLSLBase(s, t) {}
        void add_kernel(Stmt stmt,
                        const std::string &name,
                        const std::vector<DeviceArgument> &args);
    protected:

        std::string print_type(Type type, AppendSpaceIfNeeded space_option = DoNotAppendSpace);

        using CodeGen_C::visit;
        void visit(const For *);
        void visit(const Ramp *op);
        void visit(const Broadcast *op);
        void visit(const Load *op);
        void visit(const Store *op);
        void visit(const Cast *op);
        void visit(const Call *op);
        void visit(const Allocate *op);
        void visit(const Free *op);
        void visit(const Select *op);
        void visit(const Evaluate *op);
        void visit(const IntImm *op);
        void visit(const UIntImm *op);

    public:
        int workgroup_size[3];
    };

    std::ostringstream src_stream;
    std::string cur_kernel_name;
    CodeGen_OpenGLCompute_C glc;
};

}}

#endif
#ifndef HALIDE_CODEGEN_PTX_DEV_H
#define HALIDE_CODEGEN_PTX_DEV_H

/** \file
 * Defines the code-generator for producing CUDA host code
 */


namespace llvm {
class BasicBlock;
}

namespace Halide {
namespace Internal {

/** A code generator that emits GPU code from a given Halide stmt. */
class CodeGen_PTX_Dev : public CodeGen_LLVM, public CodeGen_GPU_Dev {
public:
    friend class CodeGen_GPU_Host<CodeGen_X86>;
    friend class CodeGen_GPU_Host<CodeGen_ARM>;

    /** Create a PTX device code generator. */
    CodeGen_PTX_Dev(Target host);
    ~CodeGen_PTX_Dev();

    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args);

    static void test();

    std::vector<char> compile_to_src();
    std::string get_current_kernel_name();

    void dump();

    virtual std::string print_gpu_name(const std::string &name);

    std::string api_unique_name() { return "cuda"; }

protected:
    using CodeGen_LLVM::visit;

    /** (Re)initialize the PTX module. This is separate from compile, since
     * a PTX device module will often have many kernels compiled into it for
     * a single pipeline. */
    /* override */ virtual void init_module();

    /** We hold onto the basic block at the start of the device
     * function in order to inject allocas */
    llvm::BasicBlock *entry_block;

    /** Nodes for which we need to override default behavior for the GPU runtime */
    // @{
    void visit(const For *);
    void visit(const Allocate *);
    void visit(const Free *);
    void visit(const AssertStmt *);
    // @}

    std::string march() const;
    std::string mcpu() const;
    std::string mattrs() const;
    bool use_soft_float_abi() const;
    int native_vector_bits() const;
    bool promote_indices() const {return false;}

    /** Map from simt variable names (e.g. foo.__block_id_x) to the llvm
     * ptx intrinsic functions to call to get them. */
    std::string simt_intrinsic(const std::string &name);
};

}}

#endif
#ifndef HALIDE_CODEGEN_TIRAMISU_H
#define HALIDE_CODEGEN_TIRAMISU_H

/** \file
 *
 * Defines an IRPrinter that emits C++ Tiramisu code equivalent to a halide stmt
 */

#ifndef HALIDE_SIMPLIFY_H
#define HALIDE_SIMPLIFY_H

/** \file
 * Methods for simplifying halide statements and expressions
 */

#include <cmath>


namespace Halide {
namespace Internal {

/** Perform a a wide range of simplifications to expressions and
 * statements, including constant folding, substituting in trivial
 * values, arithmetic rearranging, etc. Simplifies across let
 * statements, so must not be called on stmts with dangling or
 * repeated variable names.
 */
// @{
EXPORT Stmt simplify(Stmt, bool simplify_lets = true,
                     const Scope<Interval> &bounds = Scope<Interval>::empty_scope(),
                     const Scope<ModulusRemainder> &alignment = Scope<ModulusRemainder>::empty_scope());
EXPORT Expr simplify(Expr, bool simplify_lets = true,
                     const Scope<Interval> &bounds = Scope<Interval>::empty_scope(),
                     const Scope<ModulusRemainder> &alignment = Scope<ModulusRemainder>::empty_scope());
// @}

/** A common use of the simplifier is to prove boolean expressions are
 * true at compile time. Equivalent to is_one(simplify(e)) */
EXPORT bool can_prove(Expr e);

/** Simplify expressions found in a statement, but don't simplify
 * across different statements. This is safe to perform at an earlier
 * stage in lowering than full simplification of a stmt. */
EXPORT Stmt simplify_exprs(Stmt);

/** Implementations of division and mod that are specific to Halide.
 * Use these implementations; do not use native C division or mod to
 * simplify Halide expressions. Halide division and modulo satisify
 * the Euclidean definition of division for integers a and b:
 *
 /code
 (a/b)*b + a%b = a
 0 <= a%b < |b|
 /endcode
 *
 */
// @{
template<typename T>
inline T mod_imp(T a, T b) {
    Type t = type_of<T>();
    if (t.is_int()) {
        T r = a % b;
        r = r + (r < 0 ? (T)std::abs((int64_t)b) : 0);
        return r;
    } else {
        return a % b;
    }
}

template<typename T>
inline T div_imp(T a, T b) {
    Type t = type_of<T>();
    if (t.is_int()) {
        int64_t q = a / b;
        int64_t r = a - q * b;
        int64_t bs = b >> (t.bits() - 1);
        int64_t rs = r >> (t.bits() - 1);
        return (T) (q - (rs & bs) + (rs & ~bs));
    } else {
        return a / b;
    }
}
// @}

// Special cases for float, double.
template<> inline float mod_imp<float>(float a, float b) {
    float f = a - b * (floorf(a / b));
    // The remainder has the same sign as b.
    return f;
}
template<> inline double mod_imp<double>(double a, double b) {
    double f = a - b * (std::floor(a / b));
    return f;
}

template<> inline float div_imp<float>(float a, float b) {
    return a/b;
}
template<> inline double div_imp<double>(double a, double b) {
    return a/b;
}


EXPORT void simplify_test();

}
}

#endif


namespace Halide {

namespace Internal {

struct LoopDim {
    std::string loop_name;
    Expr min, extent;
    std::string func;
    int stage;
    std::string var;

    std::string to_string() const {
        std::ostringstream ss;
        Expr max = simplify(min + extent - 1);
        ss << min << " <= " << loop_name << " <= " << max;
        return ss.str();
    }
};

struct Stage {
    int stage;
    Expr predicate;
    std::vector<LoopDim> dims;
};

struct Computation {
    std::string name;
    // Stage -> loop dimensions
    std::map<int, Stage> stages;
    // List of all dimension names appear in the computation's stages
    std::vector<std::string> dims;
};

/** This class emits C++ code equivalent to a halide Stmt. It's
 * mostly the same as an IRPrinter, but it's wrapped in a function
 * definition, and some things are handled differently to be valid
 * C++.
 */
class CodeGen_Tiramisu : public IRVisitor {
public:
    /** Initialize a Tiramisu code generator pointing at a particular output
     * stream (e.g. a file, or std::cout) */
    CodeGen_Tiramisu(std::ostream &dest,
                     const std::string &pipeline_name,
                     const std::vector<Function> &outputs,
                     const std::vector<std::vector<int32_t>> &output_buffer_extents,
                     const std::vector<Type> &output_buffer_types,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::vector<int32_t>> &input_buffer_extents,
                     const std::vector<Type> &input_buffer_types,
                     const std::vector<std::string> &input_params,
                     const std::vector<Type> &input_param_types,
                     const std::vector<std::string> &order,
                     const std::map<std::string, Function> &env,
                     const std::map<std::pair<std::string, int>, std::vector<Dim>> &original_dim_list);
    ~CodeGen_Tiramisu();

    EXPORT std::string print(Expr e);
    EXPORT void print(Stmt s);

    EXPORT static void test();

private:
    std::string expr;
    std::ostream &stream;
    int indent;

    // The name of one Halide pipeline
    std::string pipeline;
    const std::vector<std::string> &order;
    const std::map<std::string, Function> &env;
    const std::map<std::pair<std::string, int>, std::vector<Dim>> &original_dim_list;

    std::vector<std::pair<std::string, Expr>> scope; // Scope of the variables
    std::vector<LoopDim> loop_dims; // From outermost to innermost
    std::set<std::string> output_buffers;
    std::set<std::string> input_buffers;
    std::set<std::string> temporary_buffers;
    std::set<std::string> computation_list;
    std::set<std::string> constant_list;
    std::set<std::string> extent_list;
    std::vector<std::string> buffer_str;
    std::string current_computation;

    std::string prev_computation;
    std::string prev_innermost_dim;

    std::map<std::string, std::vector<std::string>> computation_constants;
    std::map<std::string, int> duplicate_computation_count;

    std::string do_indent() const;

    void push_loop_dim(const std::string &name, Expr min, Expr extent,
                       const std::string &func_name, int stage, const std::string &var);
    void pop_loop_dim();
    std::string get_loop_bound_vars() const;
    std::string get_loop_bounds() const;
    std::string define_constant(const std::string &name, Expr value);
    std::string define_wrapper_let(const std::string &computation_name,
                                   const std::string &name, Expr value);
    void generate_schedules();
    void generate_stage_schedule(const Function &func, int stage,
                                 const StageSchedule &schedule,
                                 std::set<std::string> &vars,
                                 std::ostream &sched_ss);

    Expr substitute_in_scope(Expr expr) const;

    std::string get_current_func_name() const;
    int get_current_stage() const;
    std::string get_current_dim() const;

    void generate_buffer(const Realize *op);

    std::vector<std::string> get_stage_dims(const std::string &name, int stage, bool ignore_rvar) const;
    std::vector<std::string> get_stage_rvars(const std::string &name, int stage) const;

    template <typename T>
    void visit_binary(const T *op, const std::string &op_str);

protected:
    using IRVisitor::visit;

    void visit(const IntImm *);
    void visit(const UIntImm *);
    void visit(const FloatImm *);
    void visit(const StringImm *);
    void visit(const Cast *);
    void visit(const Variable *);
    void visit(const Add *);
    void visit(const Sub *);
    void visit(const Mul *);
    void visit(const Div *);
    void visit(const Mod *);
    void visit(const Min *);
    void visit(const Max *);
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GT *);
    void visit(const GE *);
    void visit(const And *);
    void visit(const Or *);
    void visit(const Not *);
    void visit(const Select *);
    void visit(const Load *);
    void visit(const Ramp *);
    void visit(const Broadcast *);
    void visit(const Call *);
    void visit(const Let *);
    void visit(const LetStmt *);
    void visit(const AssertStmt *);
    void visit(const ProducerConsumer *);
    void visit(const For *);
    void visit(const Store *);
    void visit(const Provide *);
    void visit(const Allocate *);
    void visit(const Free *);
    void visit(const Realize *);
    void visit(const Block *);
    void visit(const IfThenElse *);
    void visit(const Evaluate *);
    void visit(const Shuffle *);
    void visit(const Prefetch *);
};

/**
 * Dump an Tiramisu-formatted print of a Stmt to 'dest'.
 */
EXPORT void print_to_tiramisu(
    Stmt s, std::ostream &dest,
    const std::string &pipeline_name,
    const std::vector<Function> &outputs,
    const std::vector<std::vector<int32_t>> &output_buffer_extents,
    const std::vector<Type> &output_buffer_types,
    const std::vector<std::string> &inputs,
    const std::vector<std::vector<int32_t>> &input_buffer_extents,
    const std::vector<Type> &input_buffer_types,
    const std::vector<std::string> &input_params,
    const std::vector<Type> &input_param_types,
    const std::vector<std::string> &order,
    const std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_CONCISE_CASTS_H
#define HALIDE_CONCISE_CASTS_H


/** \file
 *
 * Defines concise cast and saturating cast operators to make it
 * easier to read cast-heavy code. Think carefully about the
 * readability implications before using these. They could make your
 * code better or worse. Often it's better to add extra Funcs to your
 * pipeline that do the upcasting and downcasting.
 */

namespace Halide {
namespace ConciseCasts {

inline Expr f64(Expr e) {
    return cast(Float(64, e.type().lanes()), e);
}

inline Expr f32(Expr e) {
    return cast(Float(32, e.type().lanes()), e);
}

inline Expr i64(Expr e) {
    return cast(Int(64, e.type().lanes()), e);
}

inline Expr i32(Expr e) {
    return cast(Int(32, e.type().lanes()), e);
}

inline Expr i16(Expr e) {
    return cast(Int(16, e.type().lanes()), e);
}

inline Expr i8(Expr e) {
    return cast(Int(8, e.type().lanes()), e);
}

inline Expr u64(Expr e) {
    return cast(UInt(64, e.type().lanes()), e);
}

inline Expr u32(Expr e) {
    return cast(UInt(32, e.type().lanes()), e);
}

inline Expr u16(Expr e) {
    return cast(UInt(16, e.type().lanes()), e);
}

inline Expr u8(Expr e) {
    return cast(UInt(8, e.type().lanes()), e);
}

inline Expr i8_sat(Expr e) {
    return saturating_cast(Int(8, e.type().lanes()), e);
}

inline Expr u8_sat(Expr e) {
    return saturating_cast(UInt(8, e.type().lanes()), e);
}

inline Expr i16_sat(Expr e) {
    return saturating_cast(Int(16, e.type().lanes()), e);
}

inline Expr u16_sat(Expr e) {
    return saturating_cast(UInt(16, e.type().lanes()), e);
}

inline Expr i32_sat(Expr e) {
    return saturating_cast(Int(32, e.type().lanes()), e);
}

inline Expr u32_sat(Expr e) {
    return saturating_cast(UInt(32, e.type().lanes()), e);
}

inline Expr i64_sat(Expr e) {
    return saturating_cast(Int(64, e.type().lanes()), e);
}

inline Expr u64_sat(Expr e) {
    return saturating_cast(UInt(64, e.type().lanes()), e);
}

};
};

#endif
#ifndef HALIDE_CPLUSPLUS_MANGLE_H
#define HALIDE_CPLUSPLUS_MANGLE_H

/** \file
 *
 * A simple function to get a C++ mangled function name for a function.
 */

#include <string>

namespace Halide {
namespace Internal {

/** Return the mangled C++ name for a function.
 * The target parameter is used to decide on the C++
 * ABI/mangling style to use.
 */
EXPORT std::string cplusplus_function_mangled_name(const std::string &name, const std::vector<std::string> &namespaces,
                                                   Type return_type, const std::vector<ExternFuncArgument> &args,
                                                   const Target &target);

EXPORT void cplusplus_mangle_test();

}

}

#endif
#ifndef HALIDE_INTERNAL_CSE_H
#define HALIDE_INTERNAL_CSE_H

/** \file
 * Defines a pass for introducing let expressions to wrap common sub-expressions. */


namespace Halide {
namespace Internal {

/** Replace each common sub-expression in the argument with a
 * variable, and wrap the resulting expr in a let statement giving a
 * value to that variable.
 *
 * This is important to do within Halide (instead of punting to llvm),
 * because exprs that come in from the front-end are small when
 * considered as a graph, but combinatorially large when considered as
 * a tree. For an example of a such a case, see
 * test/code_explosion.cpp */
EXPORT Expr common_subexpression_elimination(Expr);

/** Do common-subexpression-elimination on each expression in a
 * statement. Does not introduce let statements. */
EXPORT Stmt common_subexpression_elimination(Stmt);

EXPORT void cse_test();

}
}

#endif
#ifndef HALIDE_CANONICALIZE_GPU_VARS_H
#define HALIDE_CANONICALIZE_GPU_VARS_H

/** \file
 * Defines the lowering pass that canonicalize the GPU var names over.
 */


namespace Halide {
namespace Internal {

/** Canonicalize GPU var names into some pre-determined block/thread names
 * (i.e. __block_id_x, __thread_id_x, etc.). The x/y/z/w order is determined
 * by the nesting order: innermost is assigned to x and so on. */
Stmt canonicalize_gpu_vars(Stmt s);

}
}

#endif
#ifndef HALIDE_INTERNAL_DEBUG_ARGUMENTS_H
#define HALIDE_INTERNAL_DEBUG_ARGUMENTS_H

/** \file
 *
 * Defines a lowering pass that injects debug statements inside a
 * LoweredFunc. Intended to be used when Target::Debug is on.
 */

namespace Halide {
namespace Internal {

struct LoweredFunc;

/** Injects debug prints in a LoweredFunc that describe the arguments. Mutates the given func. */
void debug_arguments(LoweredFunc *func);

}
}


#endif
#ifndef HALIDE_DEBUG_TO_FILE_H
#define HALIDE_DEBUG_TO_FILE_H

/** \file
 * Defines the lowering pass that injects code at the end of
 * every realization to dump functions to a file for debugging.  */

#include <map>


namespace Halide {
namespace Internal {

/** Takes a statement with Realize nodes still unlowered. If the
 * corresponding functions have a debug_file set, then inject code
 * that will dump the contents of those functions to a file after the
 * realization. */
Stmt debug_to_file(Stmt s,
                   const std::vector<Function> &outputs,
                   const std::map<std::string, Function> &env);

}
}

#endif
#ifndef DEINTERLEAVE_H
#define DEINTERLEAVE_H

/** \file
 *
 * Defines methods for splitting up a vector into the even lanes and
 * the odd lanes. Useful for optimizing expressions such as select(x %
 * 2, f(x/2), g(x/2))
 */


namespace Halide {
namespace Internal {

/** Extract the odd-numbered lanes in a vector */
EXPORT Expr extract_odd_lanes(Expr a);

/** Extract the even-numbered lanes in a vector */
EXPORT Expr extract_even_lanes(Expr a);

/** Extract the nth lane of a vector */
EXPORT Expr extract_lane(Expr vec, int lane);

/** Look through a statement for expressions of the form select(ramp %
 * 2 == 0, a, b) and replace them with calls to an interleave
 * intrinsic */
Stmt rewrite_interleavings(Stmt s);

EXPORT void deinterleave_vector_test();

}
}

#endif
#ifndef HALIDE_EARLY_FREE_H
#define HALIDE_EARLY_FREE_H

/** \file
 * Defines the lowering pass that injects markers just after
 * the last use of each buffer so that they can potentially be freed
 * earlier.
 */


namespace Halide {
namespace Internal {

/** Take a statement with allocations and inject markers (of the form
 * of calls to "mark buffer dead") after the last use of each
 * allocation. Targets may use this to free buffers earlier than the
 * close of their Allocate node. */
Stmt inject_early_frees(Stmt s);

}
}

#endif
#ifndef HALIDE_ELF_H
#define HALIDE_ELF_H

#include <algorithm>
#include <memory>
#include <vector>
#include <list>
#include <string>
#include <iterator>
#include <limits>

namespace Halide {
namespace Internal {
namespace Elf {

// This ELF parser mostly deserializes the object into a graph
// structure in memory. It replaces indices into tables (sections,
// symbols, etc.) with a weakly referenced graph of pointers. The
// Object datastructure owns all of the objects. This namespace exists
// because it is very difficult to use LLVM's object parser to modify
// an object (it's fine for parsing only). This was built using
// http://www.skyfree.org/linux/references/ELF_Format.pdf as a reference
// for the ELF structs and constants.


class Object;
class Symbol;
class Section;
class Relocation;

// Helpful wrapper to allow range-based for loops.
template <typename T>
class iterator_range {
    T b, e;
public:
    iterator_range(T b, T e) : b(b), e(e) {}

    T begin() const { return b; }
    T end() const { return e; }
};

/** Describes a symbol */
class Symbol {
public:
    enum Binding : uint8_t {
        STB_LOCAL = 0,
        STB_GLOBAL = 1,
        STB_WEAK = 2,
        STB_LOPROC = 13,
        STB_HIPROC = 15,
    };

    enum Type : uint8_t {
        STT_NOTYPE = 0,
        STT_OBJECT = 1,
        STT_FUNC = 2,
        STT_SECTION = 3,
        STT_FILE = 4,
        STT_LOPROC = 13,
        STT_HIPROC = 15,
    };

    enum Visibility : uint8_t {
        STV_DEFAULT = 0,
        STV_INTERNAL = 1,
        STV_HIDDEN = 2,
        STV_PROTECTED = 3,
    };

private:
    std::string name;
    const Section *definition = nullptr;
    uint64_t offset = 0;
    uint32_t size = 0;
    Binding binding = STB_LOCAL;
    Type type = STT_NOTYPE;
    Visibility visibility = STV_DEFAULT;

public:
    Symbol() {}
    Symbol(const std::string &name) : name(name) {}

    /** Accesses the name of this symbol. */
    ///@{
    Symbol &set_name(const std::string &name) {
        this->name = name;
        return *this;
    }
    const std::string &get_name() const { return name; }
    ///@}

    /** Accesses the type of this symbol. */
    ///@{
    Symbol &set_type(Type type) {
        this->type = type;
        return *this;
    }
    Type get_type() const { return type; }
    ///@}

    /** Accesses the properties that describe the definition of this symbol. */
    ///@{
    Symbol &define(const Section *section, uint64_t offset, uint32_t size) {
        this->definition = section;
        this->offset = offset;
        this->size = size;
        return *this;
    }
    bool is_defined() const { return definition != nullptr; }
    const Section *get_section() const { return definition; }
    uint64_t get_offset() const { return offset; }
    uint32_t get_size() const { return size; }
    ///@}

    /** Access the binding and visibility of this symbol. See the ELF
     * spec for more information about these properties. */
    ///@{
    Symbol &set_binding(Binding binding) {
        this->binding = binding;
        return *this;
    }
    Symbol &set_visibility(Visibility visibility) {
        this->visibility = visibility;
        return *this;
    }
    Binding get_binding() const { return binding; }
    Visibility get_visibility() const { return visibility; }
    ///@}
};

/** Describes a relocation to be applied to an offset of a section in
 * an Object. */
class Relocation {
    uint32_t type = 0;
    uint64_t offset = 0;
    int64_t addend = 0;
    const Symbol *symbol = nullptr;

public:
    Relocation() {}
    Relocation(uint32_t type, uint64_t offset, int64_t addend, const Symbol *symbol)
        : type(type), offset(offset), addend(addend), symbol(symbol) {}

    /** The type of relocation to be applied. The meaning of this
     * value depends on the machine of the object. */
    ///@{
    Relocation &set_type(uint32_t type) {
        this->type = type;
        return *this;
    }
    uint32_t get_type() const { return type; }
    ///@}

    /** Where to apply the relocation. This is relative to the section
     * the relocation belongs to. */
    ///@{
    Relocation &set_offset(uint64_t offset) {
        this->offset = offset;
        return *this;
    }
    uint64_t get_offset() const { return offset; }
    ///@}

    /** The value to replace with the relocation is the address of the symbol plus the addend. */
    ///@{
    Relocation &set_symbol(const Symbol *symbol) {
        this->symbol = symbol;
        return *this;
    }
    Relocation &set_addend(int64_t addend) {
        this->addend = addend;
        return *this;
    }
    const Symbol *get_symbol() const { return symbol; }
    int64_t get_addend() const { return addend; }
    ///@}
};

/** Describes a section of an object file. */
class Section {
public:
    enum Type : uint32_t {
        SHT_NULL = 0,
        SHT_PROGBITS = 1,
        SHT_SYMTAB = 2,
        SHT_STRTAB = 3,
        SHT_RELA = 4,
        SHT_HASH = 5,
        SHT_DYNAMIC = 6,
        SHT_NOTE = 7,
        SHT_NOBITS = 8,
        SHT_REL = 9,
        SHT_SHLIB = 10,
        SHT_DYNSYM = 11,
        SHT_LOPROC = 0x70000000,
        SHT_HIPROC = 0x7fffffff,
        SHT_LOUSER = 0x80000000,
        SHT_HIUSER = 0xffffffff,
    };

    enum Flag : uint32_t {
        SHF_WRITE = 0x1,
        SHF_ALLOC = 0x2,
        SHF_EXECINSTR = 0x4,
        SHF_MASKPROC = 0xf0000000,
    };

    typedef std::vector<Relocation> RelocationList;
    typedef RelocationList::iterator relocation_iterator;
    typedef RelocationList::const_iterator const_relocation_iterator;

    typedef std::vector<char>::iterator contents_iterator;
    typedef std::vector<char>::const_iterator const_contents_iterator;

private:
    std::string name;
    Type type = SHT_NULL;
    uint32_t flags = 0;
    std::vector<char> contents;
    // Sections may have a size larger than the contents.
    uint64_t size = 0;
    uint64_t alignment = 1;
    RelocationList relocs;

public:
    Section() {}
    Section(const std::string &name, Type type) : name(name), type(type) {}

    Section &set_name(const std::string &name) {
        this->name = name;
        return *this;
    }
    const std::string &get_name() const { return name; }

    Section &set_type(Type type) {
        this->type = type;
        return *this;
    }
    Type get_type() const { return type; }

    Section &set_flag(Flag flag) {
        this->flags |= flag;
        return *this; }
    Section &remove_flag(Flag flag) {
        this->flags &= ~flag;
        return *this;
    }
    Section &set_flags(uint32_t flags) {
        this->flags = flags;
        return *this;
    }
    uint32_t get_flags() const { return flags; }
    bool is_alloc() const { return (flags & SHF_ALLOC) != 0; }
    bool is_writable() const { return (flags & SHF_WRITE) != 0; }

    /** Get or set the size of the section. The size may be larger
     * than the content. */
    ///@{
    Section &set_size(uint64_t size) {
        this->size = size;
        return *this;
    }
    uint64_t get_size() const { return std::max((uint64_t) size, (uint64_t) contents.size()); }
    ///@}

    Section &set_alignment(uint64_t alignment) {
        this->alignment = alignment;
        return *this;
    }
    uint64_t get_alignment() const { return alignment; }

    Section &set_contents(std::vector<char> contents) {
        this->contents = std::move(contents);
        return *this;
    }
    template <typename It>
    Section &set_contents(It begin, It end) {
        this->contents.assign(begin, end);
        return *this;
    }
    template <typename It>
    Section &append_contents(It begin, It end) {
        this->contents.insert(this->contents.end(), begin, end);
        return *this;
    }
    template <typename It>
    Section &prepend_contents(It begin, It end) {
        typedef typename std::iterator_traits<It>::value_type T;
        uint64_t size_bytes = std::distance(begin, end) * sizeof(T);
        this->contents.insert(this->contents.begin(), begin, end);

        // When we add data to the start of the section, we need to fix up
        // the offsets of the relocations linked to this section.
        for (Relocation &r : relocations()) {
            r.set_offset(r.get_offset() + size_bytes);
        }

        return *this;
    }
    /** Set, append or prepend an object to the contents, assuming T is a
     * trivially copyable datatype. */
    template <typename T>
    Section &set_contents(const std::vector<T> &contents) {
        this->contents.assign((const char *)contents.data(), (const char *)(contents.data() + contents.size()));
        return *this;
    }
    template <typename T>
    Section &append_contents(const T& x) {
        return append_contents((const char *)&x, (const char *)(&x + 1));
    }
    template <typename T>
    Section &prepend_contents(const T& x) {
        return prepend_contents((const char *)&x, (const char *)(&x + 1));
    }
    const std::vector<char> &get_contents() const { return contents; }
    contents_iterator contents_begin() { return contents.begin(); }
    contents_iterator contents_end() { return contents.end(); }
    const_contents_iterator contents_begin() const { return contents.begin(); }
    const_contents_iterator contents_end() const { return contents.end(); }
    const char *contents_data() const { return contents.data(); }
    size_t contents_size() const { return contents.size(); }
    bool contents_empty() const { return contents.empty(); }

    Section &set_relocations(std::vector<Relocation> relocs) {
        this->relocs = std::move(relocs);
        return *this;
    }
    template <typename It>
    Section &set_relocations(It begin, It end) {
        this->relocs.assign(begin, end);
        return *this;
    }
    void add_relocation(const Relocation &reloc) { relocs.push_back(reloc); }
    relocation_iterator relocations_begin() { return relocs.begin(); }
    relocation_iterator relocations_end() { return relocs.end(); }
    iterator_range<relocation_iterator> relocations() { return {relocs.begin(), relocs.end()}; }
    const_relocation_iterator relocations_begin() const { return relocs.begin(); }
    const_relocation_iterator relocations_end() const { return relocs.end(); }
    iterator_range<const_relocation_iterator> relocations() const { return {relocs.begin(), relocs.end()}; }
    size_t relocations_size() const { return relocs.size(); }
};

/** Base class for a target architecture to implement the target
 * specific aspects of linking. */
class Linker {
public:
    virtual ~Linker() {}

    virtual uint16_t get_machine() = 0;
    virtual uint32_t get_flags() = 0;
    virtual uint32_t get_version() = 0;
    virtual void append_dynamic(Section &dynamic) = 0;

    /** Add or get an entry to the global offset table (GOT) with a
     * relocation pointing to sym. */
    virtual uint64_t get_got_entry(Section &got, const Symbol &sym) = 0;

    /** Check to see if this relocation should go through the PLT. */
    virtual bool needs_plt_entry(const Relocation &reloc) = 0;

    /** Add a PLT entry for a symbol sym defined externally. Returns a
     * symbol representing the PLT entry. */
    virtual Symbol add_plt_entry(const Symbol &sym, Section &plt, Section &got,
                                 const Symbol &got_sym) = 0;

    /** Perform a relocation. This function may opt to not apply the
     * relocation, and return a new relocation to be performed at
     * runtime. This requires that the section to apply the relocation
     * to is writable at runtime. */
    virtual Relocation relocate(uint64_t fixup_offset, char *fixup_addr, uint64_t type,
                                const Symbol *sym, uint64_t sym_offset, int64_t addend,
                                Section &got) = 0;

};

/** Holds all of the relevant sections and symbols for an object. */
class Object {
public:
    enum Type : uint16_t {
        ET_NONE = 0,
        ET_REL = 1,
        ET_EXEC = 2,
        ET_DYN = 3,
        ET_CORE = 4,
        ET_LOPROC = 0xff00,
        ET_HIPROC = 0xffff,
    };

    // We use lists for sections and symbols to avoid iterator
    // invalidation when we modify the containers.
    typedef std::list<Section> SectionList;
    typedef typename SectionList::iterator section_iterator;
    typedef typename SectionList::const_iterator const_section_iterator;

    typedef std::list<Symbol> SymbolList;
    typedef typename SymbolList::iterator symbol_iterator;
    typedef typename SymbolList::const_iterator const_symbol_iterator;

private:
    SectionList secs;
    SymbolList syms;

    Type type = ET_NONE;
    uint16_t machine = 0;
    uint32_t version = 0;
    uint64_t entry = 0;
    uint32_t flags = 0;

    Object(const Object &);
    void operator = (const Object &);

public:
    Object() {}

    Type get_type() const { return type; }
    uint16_t get_machine() const { return machine; }
    uint32_t get_version() const { return version; }
    uint64_t get_entry() const { return entry; }
    uint32_t get_flags() const { return flags; }

    Object &set_type(Type type) {
        this->type = type;
        return *this;
    }
    Object &set_machine(uint16_t machine) {
        this->machine = machine;
        return *this;
    }
    Object &set_version(uint32_t version) {
        this->version = version;
        return *this;
    }
    Object &set_entry(uint64_t entry) {
        this->entry = entry;
        return *this;
    }
    Object &set_flags(uint32_t flags) {
        this->flags = flags;
        return *this;
    }

    /** Parse an object in memory to an Object. */
    static std::unique_ptr<Object> parse_object(const char *data, size_t size);

    /** Write a shared object in memory. */
    std::vector<char> write_shared_object(Linker *linker, const std::vector<std::string> &depedencies = {},
                                          const std::string &soname = "");

    section_iterator sections_begin() { return secs.begin(); }
    section_iterator sections_end() { return secs.end(); }
    iterator_range<section_iterator> sections() { return {secs.begin(), secs.end()}; }
    const_section_iterator sections_begin() const { return secs.begin(); }
    const_section_iterator sections_end() const { return secs.end(); }
    iterator_range<const_section_iterator> sections() const { return {secs.begin(), secs.end()}; }
    size_t sections_size() const { return secs.size(); }
    section_iterator find_section(const std::string &name);

    section_iterator add_section(const std::string &name, Section::Type type);
    section_iterator add_relocation_section(const Section &for_section);
    section_iterator erase_section(section_iterator i) { return secs.erase(i); }

    section_iterator merge_sections(const std::vector<section_iterator> &sections);
    section_iterator merge_text_sections();

    symbol_iterator symbols_begin() { return syms.begin(); }
    symbol_iterator symbols_end() { return syms.end(); }
    iterator_range<symbol_iterator> symbols() { return {syms.begin(), syms.end()}; }
    const_symbol_iterator symbols_begin() const { return syms.begin(); }
    const_symbol_iterator symbols_end() const { return syms.end(); }
    iterator_range<const_symbol_iterator> symbols() const { return {syms.begin(), syms.end()}; }
    size_t symbols_size() const { return syms.size(); }
    symbol_iterator find_symbol(const std::string &name);
    const_symbol_iterator find_symbol(const std::string &name) const;

    symbol_iterator add_symbol(const std::string &name);
};

}  // namespace Elf
}  // namespace Internal
}  // namespace Halide

#endif
#ifndef HALIDE_IR_ELIMINATE_BOOL_VECTORS_H
#define HALIDE_IR_ELIMINATE_BOOL_VECTORS_H

/** \file
 * Method to eliminate vectors of booleans from IR.
 */


namespace Halide {
namespace Internal {

/** Some targets treat vectors of bools as integers of the same type
 * that the boolean operation is being used to operate on. For
 * example, instead of select(i1x8, u16x8, u16x8), the target would
 * prefer to see select(u16x8, u16x8, u16x8), where the first argument
 * is a vector of integers representing a mask. This pass converts
 * vectors of bools to vectors of integers to meet this
 * requirement. This is done by injecting intrinsics to convert bools
 * to architecture-specific masks, and using a select_mask instrinsic
 * instead of a Select node. Because the masks are architecture
 * specific, they may not be stored or loaded. On Stores, the masks
 * are converted to UInt(8) with a value of 0 or 1, which is our
 * canonical in-memory representation of a bool. */
///@{
EXPORT Stmt eliminate_bool_vectors(Stmt s);
EXPORT Expr eliminate_bool_vectors(Expr s);
///@}

/** If a type is a boolean vector, find the type that it has been
 * changed to by eliminate_bool_vectors. */
EXPORT inline Type eliminated_bool_type(Type bool_type, Type other_type) {
    if (bool_type.is_vector() && bool_type.bits() == 1) {
        bool_type = bool_type.with_code(Type::Int).with_bits(other_type.bits());
    }
    return bool_type;
}

}  // namespace Internal
}  // namespace Halide

#endif
#ifndef HALIDE_EXPR_USES_VAR_H
#define HALIDE_EXPR_USES_VAR_H

/** \file
 * Defines a method to determine if an expression depends on some variables.
 */


namespace Halide {
namespace Internal {

template<typename T>
class ExprUsesVars : public IRGraphVisitor {
    using IRGraphVisitor::visit;

    const Scope<T> &vars;
    Scope<Expr> scope;

    void visit_name(const std::string &name) {
        if (vars.contains(name)) {
            result = true;
        } else if (scope.contains(name)) {
            include(scope.get(name));
        }
    }

    void visit(const Variable *op) {
        visit_name(op->name);
    }

    void visit(const Load *op) {
        visit_name(op->name);
        IRGraphVisitor::visit(op);
    }

    void visit(const Store *op) {
        visit_name(op->name);
        IRGraphVisitor::visit(op);
    }
public:
    ExprUsesVars(const Scope<T> &v, const Scope<Expr> *s = nullptr) : vars(v), result(false) {
        scope.set_containing_scope(s);
    }
    bool result;
};

/** Test if a statement or expression references the given variable. */
template<typename StmtOrExpr>
inline bool stmt_or_expr_uses_var(StmtOrExpr e, const std::string &v) {
    Scope<int> s;
    s.push(v, 0);
    ExprUsesVars<int> uses(s);
    e.accept(&uses);
    return uses.result;
}

/** Test if a statement or expression references any of the variables
 *  in a scope, additionally considering variables bound to Expr's in
 *  the scope provided in the final argument.
 */
template<typename StmtOrExpr, typename T>
inline bool stmt_or_expr_uses_vars(StmtOrExpr e, const Scope<T> &v,
                                   const Scope<Expr> &s = Scope<Expr>::empty_scope()) {
    ExprUsesVars<T> uses(v, &s);
    e.accept(&uses);
    return uses.result;
}

/** Test if an expression references the given variable. */
inline bool expr_uses_var(Expr e, const std::string &v) {
    return stmt_or_expr_uses_var(e, v);
}

/** Test if a statement references the given variable. */
inline bool stmt_uses_var(Stmt s, const std::string &v) {
    return stmt_or_expr_uses_var(s, v);
}

/** Test if an expression references any of the variables in a scope,
 *  additionally considering variables bound to Expr's in the scope
 *  provided in the final argument.
 */
template<typename T>
inline bool expr_uses_vars(Expr e, const Scope<T> &v,
                           const Scope<Expr> &s = Scope<Expr>::empty_scope()) {
    return stmt_or_expr_uses_vars(e, v, s);
}

/** Test if a statement references any of the variables in a scope,
 *  additionally considering variables bound to Expr's in the scope
 *  provided in the final argument.
 */
template<typename T>
inline bool stmt_uses_vars(Stmt e, const Scope<T> &v,
                           const Scope<Expr> &s = Scope<Expr>::empty_scope()) {
    return stmt_or_expr_uses_vars(e, v, s);
}

}
}

#endif
#ifndef HALIDE_EXTERN_H
#define HALIDE_EXTERN_H

/** \file
 *
 * Convenience macros that lift functions that take C types into
 * functions that take and return exprs, and call the original
 * function at runtime under the hood. See test/c_function.cpp for
 * example usage.
 */


#define _halide_check_arg_type(t, name, e, n)                     \
    _halide_user_assert(e.type() == t) << "Type mismatch for argument " << n << " to extern function " << #name << ". Type expected is " << t << " but the argument " << e << " has type " << e.type() << ".\n";

#define HalideExtern_1(rt, name, t1)                                    \
    Halide::Expr name(const Halide::Expr &a1) {                         \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1}, Halide::Internal::Call::Extern); \
    }

#define HalideExtern_2(rt, name, t1, t2)                                \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2}, Halide::Internal::Call::Extern); \
    }

#define HalideExtern_3(rt, name, t1, t2, t3)                            \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2,const Halide::Expr &a3) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2, a3}, Halide::Internal::Call::Extern); \
    }

#define HalideExtern_4(rt, name, t1, t2, t3, t4)                        \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2, const Halide::Expr &a3, const Halide::Expr &a4) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);     \
        _halide_check_arg_type(Halide::type_of<t4>(), name, a4, 4);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2, a3, a4}, Halide::Internal::Call::Extern); \
    }

#define HalideExtern_5(rt, name, t1, t2, t3, t4, t5)                    \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2, const Halide::Expr &a3, const Halide::Expr &a4, const Halide::Expr &a5) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);     \
        _halide_check_arg_type(Halide::type_of<t4>(), name, a4, 4);     \
        _halide_check_arg_type(Halide::type_of<t5>(), name, a5, 5);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2, a3, a4, a5}, Halide::Internal::Call::Extern); \
    }

#define HalidePureExtern_1(rt, name, t1)                                \
    Halide::Expr name(const Halide::Expr &a1) {                         \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1}, Halide::Internal::Call::PureExtern); \
    }

#define HalidePureExtern_2(rt, name, t1, t2)                            \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2}, Halide::Internal::Call::PureExtern); \
    }

#define HalidePureExtern_3(rt, name, t1, t2, t3)                        \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2, const Halide::Expr &a3) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2, a3}, Halide::Internal::Call::PureExtern); \
    }

#define HalidePureExtern_4(rt, name, t1, t2, t3, t4)                    \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2, const Halide::Expr &a3, const Halide::Expr &a4) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);     \
        _halide_check_arg_type(Halide::type_of<t4>(), name, a4, 4);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2, a3, a4}, Halide::Internal::Call::PureExtern); \
    }

#define HalidePureExtern_5(rt, name, t1, t2, t3, t4, t5)                \
    Halide::Expr name(const Halide::Expr &a1, const Halide::Expr &a2, const Halide::Expr &a3, const Halide::Expr &a4, const Halide::Expr &a5) { \
        _halide_check_arg_type(Halide::type_of<t1>(), name, a1, 1);     \
        _halide_check_arg_type(Halide::type_of<t2>(), name, a2, 2);     \
        _halide_check_arg_type(Halide::type_of<t3>(), name, a3, 3);     \
        _halide_check_arg_type(Halide::type_of<t4>(), name, a4, 4);     \
        _halide_check_arg_type(Halide::type_of<t5>(), name, a5, 5);     \
        return Halide::Internal::Call::make(Halide::type_of<rt>(), #name, {a1, a2, a3, a4, a5}, Halide::Internal::Call::PureExtern); \
    }
#endif
#ifndef HALIDE_FAST_INTEGER_DIVIDE_H
#define HALIDE_FAST_INTEGER_DIVIDE_H


namespace Halide {

/** Built-in images used for fast_integer_divide below. Use of
 * fast_integer_divide will automatically embed the appropriate tables
 * in your object file. They are declared here in case you want to do
 * something non-default with them. */
namespace IntegerDivideTable {
EXPORT Buffer<uint8_t> integer_divide_table_u8();
EXPORT Buffer<uint8_t> integer_divide_table_s8();
EXPORT Buffer<uint16_t> integer_divide_table_u16();
EXPORT Buffer<uint16_t> integer_divide_table_s16();
EXPORT Buffer<uint32_t> integer_divide_table_u32();
EXPORT Buffer<uint32_t> integer_divide_table_s32();
}


/** Integer division by small values can be done exactly as multiplies
 * and shifts. This function does integer division for numerators of
 * various integer types (8, 16, 32 bit signed and unsigned)
 * numerators and uint8 denominators. The type of the result is the
 * type of the numerator. The unsigned version is faster than the
 * signed version, so cast the numerator to an unsigned int if you
 * know it's positive.
 *
 * If your divisor is compile-time constant, Halide performs a
 * slightly better optimization automatically, so there's no need to
 * use this function (but it won't hurt).
 *
 * This function vectorizes well on arm, and well on x86 for 16 and 8
 * bit vectors. For 32-bit vectors on x86 you're better off using
 * native integer division.
 *
 * Also, this routine treats division by zero as division by
 * 256. I.e. it interprets the uint8 divisor as a number from 1 to 256
 * inclusive.
 */
EXPORT Expr fast_integer_divide(Expr numerator, Expr denominator);

/** Use the fast integer division tables to implement a modulo
 * operation via the Euclidean identity: a%b = a - (a/b)*b
 */
EXPORT Expr fast_integer_modulo(Expr numerator, Expr denominator);

}

#endif
#ifndef FIND_CALLS_H
#define FIND_CALLS_H

/** \file
 *
 * Defines analyses to extract the functions called a function.
 */

#include <map>


namespace Halide {
namespace Internal {

/** Construct a map from name to Function definition object for all Halide
 *  functions called directly in the definition of the Function f, including
 *  in update definitions, update index expressions, and RDom extents. This map
 *  _does not_ include the Function f, unless it is called recursively by
 *  itself.
 */
std::map<std::string, Function> find_direct_calls(Function f);

/** Construct a map from name to Function definition object for all Halide
 *  functions called directly in the definition of the Function f, or
 *  indirectly in those functions' definitions, recursively. This map always
 *  _includes_ the Function f.
 */
std::map<std::string, Function> find_transitive_calls(Function f);

/** Find all Functions transitively referenced by f in any way and add
 * them to the given map. */
void populate_environment(Function f, std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_SYNCTHREADS_H
#define HALIDE_SYNCTHREADS_H

/** \file
 * Defines the lowering pass that fuses and normalizes loops over gpu
 * threads to target CUDA, OpenCL, and Metal.
 */


namespace Halide {
namespace Internal {

/** Rewrite all GPU loops to have a min of zero. */
Stmt zero_gpu_loop_mins(Stmt s);

/** Converts Halide's GPGPU IR to the OpenCL/CUDA/Metal model. Within every
 * loop over gpu block indices, fuse the inner loops over thread
 * indices into a single loop (with predication to turn off
 * threads). Also injects synchronization points as needed, and hoists
 * allocations at the block level out into a single shared memory
 * array. */
Stmt fuse_gpu_thread_loops(Stmt s);

}
}

#endif
#ifndef FUZZ_FLOAT_STORES_H
#define FUZZ_FLOAT_STORES_H


/** \file
 * Defines a lowering pass that messes with floating point stores.
 */

namespace Halide {
namespace Internal {

/** On every store of a floating point value, mask off the
 * least-significant-bit of the mantissa. We've found that whether or
 * not this dramatically changes the output of a pipeline correlates
 * very well with whether or not a pipeline will produce very
 * different outputs on different architectures (e.g. with and without
 * FMA). It's also a useful way to detect bad tests, such as those
 * that expect exact floating point equality across platforms. */
Stmt fuzz_float_stores(Stmt s);

}
}

#endif
#ifndef HALIDE_GENERATOR_H_
#define HALIDE_GENERATOR_H_

/** \file
 *
 * Generator is a class used to encapsulate the building of Funcs in user
 * pipelines. A Generator is agnostic to JIT vs AOT compilation; it can be used for
 * either purpose, but is especially convenient to use for AOT compilation.
 *
 * A Generator explicitly declares the Inputs and Outputs associated for a given
 * pipeline, and (optionally) separates the code for constructing the outputs from the code from
 * scheduling them. For instance:
 *
 * \code
 *     class Blur : public Generator<Blur> {
 *     public:
 *         Input<Func> input{"input", UInt(16), 2};
 *         Output<Func> output{"output", UInt(16), 2};
 *         void generate() {
 *             blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
 *             blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;
 *             output(x, y) = blur(x, y);
 *         }
 *         void schedule() {
 *             blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
 *             blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);
 *         }
 *     private:
 *         Var x, y, xi, yi;
 *         Func blur_x, blur_y;
 *     };
 * \endcode
 *
 * Halide can compile a Generator into the correct pipeline by introspecting these
 * values and constructing an appropriate signature based on them.
 *
 * A Generator provides implementations of two methods:
 *
 *   - generate(), which must fill in all Output Func(s); it may optionally also do scheduling
 *   if no schedule() method is present.
 *   - schedule(), which (if present) should contain all scheduling code.
 *
 * Inputs can be any C++ scalar type:
 *
 * \code
 *     Input<float> radius{"radius"};
 *     Input<int32_t> increment{"increment"};
 * \endcode
 *
 * An Input<Func> is (essentially) like an ImageParam, except that it may (or may
 * not) not be backed by an actual buffer, and thus has no defined extents.
 *
 * \code
 *     Input<Func> input{"input", Float(32), 2};
 * \endcode
 *
 * You can optionally make the type and/or dimensions of Input<Func> unspecified,
 * in which case the value is simply inferred from the actual Funcs passed to them.
 * Of course, if you specify an explicit Type or Dimension, we still require the
 * input Func to match, or a compilation error results.
 *
 * \code
 *     Input<Func> input{ "input", 3 };  // require 3-dimensional Func,
 *                                       // but leave Type unspecified
 * \endcode
 *
 * A Generator must explicitly list the output(s) it produces:
 *
 * \code
 *     Output<Func> output{"output", Float(32), 2};
 * \endcode
 *
 * You can specify an output that returns a Tuple by specifying a list of Types:
 *
 * \code
 *     class Tupler : Generator<Tupler> {
 *       Input<Func> input{"input", Int(32), 2};
 *       Output<Func> output{"output", {Float(32), UInt(8)}, 2};
 *       void generate() {
 *         Var x, y;
 *         Expr a = cast<float>(input(x, y));
 *         Expr b = cast<uint8_t>(input(x, y));
 *         output(x, y) = Tuple(a, b);
 *       }
 *     };
 * \endcode
 *
 * You can also specify Output<X> for any scalar type (except for Handle types);
 * this is merely syntactic sugar on top of a zero-dimensional Func, but can be
 * quite handy, especially when used with multiple outputs:
 *
 * \code
 *     Output<float> sum{"sum"};  // equivalent to Output<Func> {"sum", Float(32), 0}
 * \endcode
 *
 * As with Input<Func>, you can optionally make the type and/or dimensions of an
 * Output<Func> unspecified; any unspecified types must be resolved via an
 * implicit GeneratorParam in order to use top-level compilation.
 *
 * You can also declare an *array* of Input or Output, by using an array type
 * as the type parameter:
 *
 * \code
 *     // Takes exactly 3 images and outputs exactly 3 sums.
 *     class SumRowsAndColumns : Generator<SumRowsAndColumns> {
 *       Input<Func[3]> inputs{"inputs", Float(32), 2};
 *       Input<int32_t[2]> extents{"extents"};
 *       Output<Func[3]> sums{"sums", Float(32), 1};
 *       void generate() {
 *         assert(inputs.size() == sums.size());
 *         // assume all inputs are same extent
 *         Expr width = extent[0];
 *         Expr height = extent[1];
 *         for (size_t i = 0; i < inputs.size(); ++i) {
 *           RDom r(0, width, 0, height);
 *           sums[i]() = 0.f;
 *           sums[i]() += inputs[i](r.x, r.y);
 *          }
 *       }
 *     };
 * \endcode
 *
 * You can also leave array size unspecified, with some caveats:
 *   - For ahead-of-time compilation, Inputs must have a concrete size specified
 *     via a GeneratorParam at build time (e.g., pyramid.size=3)
 *   - For JIT compilation via a Stub, Inputs array sizes will be inferred
 *     from the vector passed.
 *   - For ahead-of-time compilation, Outputs may specify a concrete size
 *     via a GeneratorParam at build time (e.g., pyramid.size=3), or the
 *     size can be specified via a resize() method.
 *
 * \code
 *     class Pyramid : public Generator<Pyramid> {
 *     public:
 *         GeneratorParam<int32_t> levels{"levels", 10};
 *         Input<Func> input{ "input", Float(32), 2 };
 *         Output<Func[]> pyramid{ "pyramid", Float(32), 2 };
 *         void generate() {
 *             pyramid.resize(levels);
 *             pyramid[0](x, y) = input(x, y);
 *             for (int i = 1; i < pyramid.size(); i++) {
 *                 pyramid[i](x, y) = (pyramid[i-1](2*x, 2*y) +
 *                                    pyramid[i-1](2*x+1, 2*y) +
 *                                    pyramid[i-1](2*x, 2*y+1) +
 *                                    pyramid[i-1](2*x+1, 2*y+1))/4;
 *             }
 *         }
 *     };
 * \endcode
 *
 * A Generator can also be customized via compile-time parameters (GeneratorParams
 * or ScheduleParams), which affect code generation.
 *
 * GeneratorParams, ScheduleParams, Inputs, and Outputs are (by convention) always
 * public and always declared at the top of the Generator class, in the order
 *
 * \code
 *     GeneratorParam(s)
 *     ScheduleParam(s)
 *     Input<Func>(s)
 *     Input<non-Func>(s)
 *     Output<Func>(s)
 * \endcode
 *
 * Note that the Inputs and Outputs will appear in the C function call in the order
 * they are declared. All Input<Func> and Output<Func> are represented as buffer_t;
 * all other Input<> are the appropriate C++ scalar type. (GeneratorParams are
 * always referenced by name, not position, so their order is irrelevant.)
 *
 * All Inputs and Outputs must have explicit names, and all such names must match
 * the regex [A-Za-z][A-Za-z_0-9]* (i.e., essentially a C/C++ variable name, with
 * some extra restrictions on underscore use). By convention, the name should match
 * the member-variable name.
 *
 * Generators are added to a global registry to simplify AOT build mechanics; this
 * is done by simply using the HALIDE_REGISTER_GENERATOR macro at global scope:
 *
 * \code
 *      HALIDE_REGISTER_GENERATOR(ExampleGen, jit_example)
 * \endcode
 *
 * The registered name of the Generator is provided must match the same rules as
 * Input names, above.
 *
 * Note that the class name of the generated Stub class will match the registered
 * name by default; if you want to vary it (typically, to include namespaces),
 * you can add it as an optional third argument:
 *
 * \code
 *      HALIDE_REGISTER_GENERATOR(ExampleGen, jit_example, SomeNamespace::JitExampleStub)
 * \endcode
 *
 * Note that a Generator is always executed with a specific Target assigned to it,
 * that you can access via the get_target() method. (You should *not* use the
 * global get_target_from_environment(), etc. methods provided in Target.h)
 *
 * (Note that there are older variations of Generator that differ from what's
 * documented above; these are still supported but not described here. See
 * https://github.com/halide/Halide/wiki/Old-Generator-Documentation for
 * more information.)
 */

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifndef HALIDE_IMAGE_PARAM_H
#define HALIDE_IMAGE_PARAM_H

/** \file
 *
 * Classes for declaring image parameters to halide pipelines
 */


namespace Halide {

/** An Image parameter to a halide pipeline. E.g., the input image. */
class ImageParam : public OutputImageParam {

    /** Func representation of the ImageParam.
     * All call to ImageParam is equivalent to call to its intrinsic Func
     * representation. */
    Func func;

    /** Helper function to initialize the Func representation of this ImageParam. */
    EXPORT void init_func();

public:

    /** Construct a nullptr image parameter handle. */
    ImageParam() : OutputImageParam() {}

    /** Construct an image parameter of the given type and
     * dimensionality, with an auto-generated unique name. */
    EXPORT ImageParam(Type t, int d);

    /** Construct an image parameter of the given type and
     * dimensionality, with the given name */
    EXPORT ImageParam(Type t, int d, const std::string &n);

    /** Bind an Image to this ImageParam. Only relevant for jitting */
    // @{
    EXPORT void set(Buffer<> im);
    // @}

    /** Get a reference to the Buffer bound to this ImageParam. Only relevant for jitting. */
    // @{
    EXPORT Buffer<> get() const;
    // @}

    /** Unbind any bound Buffer */
    EXPORT void reset();

    /** Construct an expression which loads from this image
     * parameter. The location is extended with enough implicit
     * variables to match the dimensionality of the image
     * (see \ref Var::implicit)
     */
    // @{
    template <typename... Args>
    NO_INLINE Expr operator()(Args&&... args) const {
        return func(std::forward<Args>(args)...);
    }
    EXPORT Expr operator()(std::vector<Expr>) const;
    EXPORT Expr operator()(std::vector<Var>) const;
    // @}

    /** Return the intrinsic Func representation of this ImageParam. This allows
     * an ImageParam to be implicitly converted to a Func.
     *
     * Note that we use implicit vars to name the dimensions of Funcs associated
     * with the ImageParam: both its internal Func representation and wrappers
     * (See \ref ImageParam::in). For example, to unroll the first and second
     * dimensions of the associated Func by a factor of 2, we would do the following:
     \code
     func.unroll(_0, 2).unroll(_1, 2);
     \endcode
     * '_0' represents the first dimension of the Func, while _1 represents the
     * second dimension of the Func.
     */
    EXPORT operator Func() const;


    /** Creates and returns a new Func that wraps this ImageParam. During
     * compilation, Halide will replace calls to this ImageParam with calls
     * to the wrapper as appropriate. If this ImageParam is already wrapped
     * for use in some Func, it will return the existing wrapper.
     *
     * For example, img.in(g) would rewrite a pipeline like this:
     \code
     ImageParam img(Int(32), 2);
     Func g;
     g(x, y) = ... img(x, y) ...
     \endcode
     * into a pipeline like this:
     \code
     ImageParam img(Int(32), 2);
     Func img_wrap, g;
     img_wrap(x, y) = img(x, y);
     g(x, y) = ... img_wrap(x, y) ...
     \endcode
     *
     * This has a variety of uses. One use case is to stage loads from an
     * ImageParam via some intermediate buffer (e.g. on the stack or in shared
     * GPU memory).
     *
     * The following example illustrates how you would use the 'in()' directive
     * to stage loads from an ImageParam into the GPU shared memory:
     \code
     ImageParam img(Int(32), 2);
     output(x, y) = img(y, x);
     Var tx, ty;
     output.compute_root().gpu_tile(x, y, tx, ty, 8, 8);
     img.in().compute_at(output, x).unroll(_0, 2).unroll(_1, 2).gpu_threads(_0, _1);
     \endcode
     *
     * Note that we use implicit vars to name the dimensions of the wrapper Func.
     * See \ref Func::in for more possible use cases of the 'in()' directive.
     */
    // @{
    EXPORT Func in(const Func &f);
    EXPORT Func in(const std::vector<Func> &fs);
    EXPORT Func in();
    // @}
};

}

#endif
#ifndef HALIDE_OBJECT_INSTANCE_REGISTRY_H
#define HALIDE_OBJECT_INSTANCE_REGISTRY_H

/** \file
 *
 * Provides a single global registry of Generators, GeneratorParams,
 * and Params indexed by this pointer. This is used for finding the
 * parameters inside of a Generator. NOTE: this is threadsafe only
 * if you are compiling with C++11 enabled.
 */

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <mutex>
#include <vector>

namespace Halide {
namespace Internal {

class ObjectInstanceRegistry {
public:
    enum Kind {
        Invalid,
        Generator,
        GeneratorParam,
        GeneratorInput,
        GeneratorOutput,
        FilterParam,
        ScheduleParam
    };

    /** Add an instance to the registry. The size may be 0 for Param Kinds,
     * but not for Generator. subject_ptr is the value actually associated
     * with this instance; it is usually (but not necessarily) the same
     * as this_ptr. Assert if this_ptr is already registered.
     *
     * If 'this' is directly heap allocated (not a member of a
     * heap-allocated object) and you want the introspection subsystem
     * to know about it and its members, set the introspection_helper
     * argument to a pointer to a global variable with the same true
     * type as 'this'. For example:
     *
     * MyObject *obj = new MyObject;
     * static MyObject *introspection_helper = nullptr;
     * register_instance(obj, sizeof(MyObject), kind, obj, &introspection_helper);
     *
     * I.e. introspection_helper should be a pointer to a pointer to
     * an object instance. The inner pointer can be null. The
     * introspection subsystem will then assume this new object is of
     * the matching type, which will help its members deduce their
     * names on construction.
     */
    static void register_instance(void *this_ptr, size_t size, Kind kind, void *subject_ptr,
                                  const void *introspection_helper);

    /** Remove an instance from the registry. Assert if not found.
     */
    static void unregister_instance(void *this_ptr);

    /** Returns the list of subject pointers for objects that have
     *   been directly registered within the given range. If there is
     *   another containing object inside the range, instances within
     *   that object are skipped.
     */
    static std::vector<void *> instances_in_range(void *start, size_t size, Kind kind);

private:
    static ObjectInstanceRegistry &get_registry();

    struct InstanceInfo {
        void *subject_ptr;  // May be different from the this_ptr in the key
        size_t size;  // May be 0 for params
        Kind kind;
        bool registered_for_introspection;

        InstanceInfo() : subject_ptr(nullptr), size(0), kind(Invalid), registered_for_introspection(false) {}
        InstanceInfo(size_t size, Kind kind, void *subject_ptr, bool registered_for_introspection)
            : subject_ptr(subject_ptr), size(size), kind(kind), registered_for_introspection(registered_for_introspection) {}
    };

    std::mutex mutex;
    std::map<uintptr_t, InstanceInfo> instances;

    ObjectInstanceRegistry() {}
    ObjectInstanceRegistry(ObjectInstanceRegistry &rhs);  // unimplemented
};

}  // namespace Internal
}  // namespace Halide

#endif  // HALIDE_OBJECT_INSTANCE_REGISTRY_H
#ifndef HALIDE_SCHEDULE_PARAM_H
#define HALIDE_SCHEDULE_PARAM_H

#include <type_traits>


/** \file
 *
 * Classes for declaring scalar parameters to halide pipelines
 */

namespace Halide {

namespace Internal {

class GeneratorBase;

class ScheduleParamBase {
public:
    const std::string &name() const {
        return sp_name;
    }

    bool is_looplevel_param() const {
        return type == Handle();
    }

    const Type &scalar_type() const {
        internal_assert(!is_looplevel_param());
        return type;
    }

    operator Expr() const {
        user_assert(!is_looplevel_param()) << "Only scalar ScheduleParams can be converted to Expr.";
        return scalar_expr;
    }

    operator LoopLevel() const {
        user_assert(is_looplevel_param()) << "Only ScheduleParam<LoopLevel> can be converted to LoopLevel.";
        return loop_level;
    }

    // overload the set() function to call the right virtual method based on type.
    // This allows us to attempt to set a ScheduleParam via a
    // plain C++ type, even if we don't know the specific templated
    // subclass. Attempting to set the wrong type will assert.
    //
    // It's always a bit iffy to use macros for this, but IMHO it clarifies the situation here.
#define HALIDE_SCHEDULE_PARAM_TYPED_SETTER(TYPE) \
    virtual void set(const TYPE &new_value) = 0;

    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(bool)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int8_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int16_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int32_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int64_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint8_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint16_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint32_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint64_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(float)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(double)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(LoopLevel)

#undef HALIDE_SCHEDULE_PARAM_TYPED_SETTER

protected:
    friend class GeneratorBase;

    std::string sp_name;
    Type type;
    Internal::Parameter scalar_parameter;
    Expr scalar_expr;
    LoopLevel loop_level;

    EXPORT ScheduleParamBase(const Type &t, const std::string &name, bool is_explicit_name);
    EXPORT virtual ~ScheduleParamBase();

    // This is provided only for GeneratorBase; other code should not need to use it.
    virtual void set_from_string(const std::string &new_value_string) = 0;

    EXPORT explicit ScheduleParamBase(const ScheduleParamBase &);
    EXPORT ScheduleParamBase &operator=(const ScheduleParamBase &);
};

}  // namespace Internal

// This is strictly some syntactic sugar to suppress certain compiler warnings.
template<typename FROM, typename TO>
struct Convert {
    template <typename TO2 = TO, typename std::enable_if<!std::is_same<TO2, bool>::value>::type * = nullptr>
    inline static TO2 value(const FROM &from) { return static_cast<TO2>(from); }

    template <typename TO2 = TO, typename std::enable_if<std::is_same<TO2, bool>::value>::type * = nullptr>
    inline static TO2 value(const FROM &from) { return from != 0; }
};

/** A ScheduleParam is a "Param" that can contain a scalar Expr or a LoopLevel;
 * unlike Param<>, its value cannot be set at runtime. All ScheduleParam values
 * are finalized just before lowering, and must translate into a constant scalar
 * value (or a well-defined LoopLevel) at that point. The value of
 * should be bound to an actual value of type T using the set method
 * before you realize the function uses this. If you're statically
 * compiling, this param should *not* appear in the argument list.
 */
template <typename T>
class ScheduleParam : public Internal::ScheduleParamBase {
    using ScheduleParamBase = Internal::ScheduleParamBase;

    template <typename T2 = T,
              typename std::enable_if<std::is_arithmetic<T2>::value>::type * = nullptr>
    static Type get_param_type() {
        return type_of<T>();
    }

    template <typename T2 = T,
              typename std::enable_if<!std::is_arithmetic<T2>::value>::type * = nullptr>
    static Type get_param_type() {
        return Handle();
    }

    template <typename T2, typename std::enable_if<std::is_arithmetic<T>::value &&
                                                   std::is_convertible<T2, T>::value>::type * = nullptr>
    HALIDE_ALWAYS_INLINE void typed_setter_impl(const T2 &value, const char *type) {
        // Arithmetic types must roundtrip losslessly.
        if (!std::is_same<T, T2>::value &&
            std::is_arithmetic<T>::value &&
            std::is_arithmetic<T2>::value) {
            const T t = Convert<T2, T>::value(value);
            const T2 t2 = Convert<T, T2>::value(t);
            if (t2 != value) {
                user_error << "The ScheduleParam " << name() << " cannot be set with a value of type " << type << ".\n";
            }
        }
        scalar_parameter.set_scalar<T>(Convert<T2, T>::value(value));
    }

    template <typename T2, typename std::enable_if<std::is_same<T2, LoopLevel>::value>::type * = nullptr>
    HALIDE_ALWAYS_INLINE void typed_setter_impl(const LoopLevel &value, const char *msg) {
        user_assert(is_looplevel_param()) << "Only ScheduleParam<LoopLevel> can be set withLoopLevel.";
        loop_level.copy_from(value);
    }

    template <typename T2, typename std::enable_if<!std::is_convertible<T2, T>::value>::type * = nullptr>
    HALIDE_ALWAYS_INLINE void typed_setter_impl(const T2 &value, const char *type) {
        user_error << "The ScheduleParam " << name() << " cannot be set with a value of type " << type << ".\n";
    }

    template <typename T2 = T,
              typename std::enable_if<std::is_same<T2, LoopLevel>::value>::type * = nullptr>
    NO_INLINE void set_from_string_impl(const std::string &new_value_string) {
        if (new_value_string == "root") {
            set(LoopLevel::root());
        } else if (new_value_string == "inline") {
            set(LoopLevel::inlined());
        } else {
            user_error << "Unable to parse " << name() << ": " << new_value_string;
        }
    }

    template <typename T2 = T,
              typename std::enable_if<std::is_same<T2, bool>::value>::type * = nullptr>
    NO_INLINE void set_from_string_impl(const std::string &new_value_string) {
        if (new_value_string == "true") {
            set(true);
        } else if (new_value_string == "false") {
            set(false);
        } else {
            user_error << "Unable to parse " << name() << ": " << new_value_string;
        }
    }

    template <typename T2 = T,
              typename std::enable_if<std::is_arithmetic<T2>::value && !std::is_same<T2, bool>::value>::type * = nullptr>
    NO_INLINE void set_from_string_impl(const std::string &new_value_string) {
        std::istringstream iss(new_value_string);
        T t;
        iss >> t;
        user_assert(!iss.fail() && iss.get() == EOF) << "Unable to parse " << name() << ": " << new_value_string;
        set(t);
    }

protected:
    void set_from_string(const std::string &new_value_string) override {
        set_from_string_impl(new_value_string);
    }

public:
    ScheduleParam() : ScheduleParamBase(get_param_type(), "", false) {}

    explicit ScheduleParam(const std::string &name) : ScheduleParamBase(get_param_type(), name, true) {}

    ScheduleParam(const std::string &name, const T &value) : ScheduleParamBase(get_param_type(), name, true) {
        set(value);
    }

    // TODO hide?
    explicit ScheduleParam(const ScheduleParamBase &that) : ScheduleParamBase(that) {}

#define HALIDE_SCHEDULE_PARAM_TYPED_SETTER(TYPE) \
    void set(const TYPE &new_value) override { typed_setter_impl<TYPE>(new_value, #TYPE); }

    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(bool)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int8_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int16_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int32_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(int64_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint8_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint16_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint32_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(uint64_t)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(float)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(double)
    HALIDE_SCHEDULE_PARAM_TYPED_SETTER(LoopLevel)

#undef HALIDE_SCHEDULE_PARAM_TYPED_SETTER

    // Note that we deliberately do not provide a way to retrieve the non-Expr value
    // of ScheduleParam: this is because the value is probably inaccurate at the point
    // you'd be tempted to examine it, since it won't be finalized until the start of lowering.
    // Here's the code that we'd use to do so, if we find we need to:

    // template <typename T2 = T,
    //           typename std::enable_if<std::is_arithmetic<T2>::value>::type * = nullptr>
    // operator T() const {
    //     return scalar_parameter.get_scalar<T>();
    // }

    // template <typename T2 = T,
    //           typename std::enable_if<std::is_same<T2, LoopLevel>::value>::type * = nullptr>
    // operator T() const {
    //     return loop_level;
    // }
};

}  // namespace Halide

#endif

namespace Halide {

template<typename T> class Buffer;

namespace Internal {

EXPORT void generator_test();

/**
 * ValueTracker is an internal utility class that attempts to track and flag certain
 * obvious Stub-related errors at Halide compile time: it tracks the constraints set
 * on any Parameter-based argument (i.e., Input<Buffer> and Output<Buffer>) to
 * ensure that incompatible values aren't set.
 *
 * e.g.: if a Generator A requires stride[0] == 1,
 * and Generator B uses Generator A via stub, but requires stride[0] == 4,
 * we should be able to detect this at Halide compilation time, and fail immediately,
 * rather than producing code that fails at runtime and/or runs slowly due to
 * vectorization being unavailable.
 *
 * We do this by tracking the active values at entrance and exit to all user-provided
 * Generator methods (build()/generate()/schedule()); if we ever find more than two unique
 * values active, we know we have a potential conflict. ("two" here because the first
 * value is the default value for a given constraint.)
 *
 * Note that this won't catch all cases:
 * -- JIT compilation has no way to check for conflicts at the top-level
 * -- constraints that match the default value (e.g. if dim(0).set_stride(1) is the
 * first value seen by the tracker) will be ignored, so an explicit requirement set
 * this way can be missed
 *
 * Nevertheless, this is likely to be much better than nothing when composing multiple
 * layers of Stubs in a single fused result.
 */
class ValueTracker {
private:
    std::map<std::string, std::vector<std::vector<Expr>>> values_history;
    const size_t max_unique_values;
public:
    explicit ValueTracker(size_t max_unique_values = 2) : max_unique_values(max_unique_values) {}
    EXPORT void track_values(const std::string &name, const std::vector<Expr> &values);
};

EXPORT std::vector<Expr> parameter_constraints(const Parameter &p);

template <typename T>
NO_INLINE std::string enum_to_string(const std::map<std::string, T> &enum_map, const T& t) {
    for (auto key_value : enum_map) {
        if (t == key_value.second) {
            return key_value.first;
        }
    }
    user_error << "Enumeration value not found.\n";
    return "";
}

template <typename T>
T enum_from_string(const std::map<std::string, T> &enum_map, const std::string& s) {
    auto it = enum_map.find(s);
    user_assert(it != enum_map.end()) << "Enumeration value not found: " << s << "\n";
    return it->second;
}

EXPORT extern const std::map<std::string, Halide::Type> &get_halide_type_enum_map();
inline std::string halide_type_to_enum_string(const Type &t) {
    return enum_to_string(get_halide_type_enum_map(), t);
}

EXPORT extern const std::map<std::string, Halide::LoopLevel> &get_halide_looplevel_enum_map();
inline std::string halide_looplevel_to_enum_string(const LoopLevel &loop_level){
    return enum_to_string(get_halide_looplevel_enum_map(), loop_level);
}

// Convert a Halide Type into a string representation of its C source.
// e.g., Int(32) -> "Halide::Int(32)"
EXPORT std::string halide_type_to_c_source(const Type &t);

// Convert a Halide Type into a string representation of its C Source.
// e.g., Int(32) -> "int32_t"
EXPORT std::string halide_type_to_c_type(const Type &t);

/** generate_filter_main() is a convenient wrapper for GeneratorRegistry::create() +
 * compile_to_files(); it can be trivially wrapped by a "real" main() to produce a
 * command-line utility for ahead-of-time filter compilation. */
EXPORT int generate_filter_main(int argc, char **argv, std::ostream &cerr);

// select_type<> is to std::conditional as switch is to if:
// it allows a multiway compile-time type definition via the form
//
//    select_type<cond<condition1, type1>,
//                cond<condition2, type2>,
//                ....
//                cond<conditionN, typeN>>::type
//
// Note that the conditions are evaluated in order; the first evaluating to true
// is chosen.
//
// Note that if no conditions evaluate to true, the resulting type is illegal
// and will produce a compilation error. (You can provide a default by simply
// using cond<true, SomeType> as the final entry.)
template<bool B, typename T>
struct cond {
    static constexpr bool value = B;
    using type = T;
};

template <typename First, typename... Rest>
struct select_type : std::conditional<First::value, typename First::type, typename select_type<Rest...>::type> { };

template<typename First>
struct select_type<First> { using type = typename std::conditional<First::value, typename First::type, void>::type; };

class GeneratorBase;

class GeneratorParamBase {
public:
    EXPORT explicit GeneratorParamBase(const std::string &name);
    EXPORT virtual ~GeneratorParamBase();

    const std::string name;

    // overload the set() function to call the right virtual method based on type.
    // This allows us to attempt to set a GeneratorParam via a
    // plain C++ type, even if we don't know the specific templated
    // subclass. Attempting to set the wrong type will assert.
    // Notice that there is no typed setter for Enums, for obvious reasons;
    // setting enums in an unknown type must fallback to using set_from_string.
    //
    // It's always a bit iffy to use macros for this, but IMHO it clarifies the situation here.
#define HALIDE_GENERATOR_PARAM_TYPED_SETTER(TYPE) \
    virtual void set(const TYPE &new_value) = 0;

    HALIDE_GENERATOR_PARAM_TYPED_SETTER(bool)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int8_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int16_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int32_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int64_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint8_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint16_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint32_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint64_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(float)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(double)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(Target)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(Type)

#undef HALIDE_GENERATOR_PARAM_TYPED_SETTER

    // Add overloads for string and char*
    void set(const std::string &new_value) { set_from_string(new_value); }
    void set(const char *new_value) { set_from_string(std::string(new_value)); }

protected:
    friend class GeneratorBase;
    friend class StubEmitter;

    EXPORT void check_value_readable() const;
    EXPORT void check_value_writable() const;

    // All GeneratorParams are settable from string.
    virtual void set_from_string(const std::string &value_string) = 0;

    virtual std::string to_string() const = 0;
    virtual std::string call_to_string(const std::string &v) const = 0;
    virtual std::string get_c_type() const = 0;

    virtual std::string get_type_decls() const {
        return "";
    }

    virtual std::string get_default_value() const {
        return to_string();
    }

    virtual std::string get_template_type() const {
        return get_c_type();
    }

    virtual std::string get_template_value() const {
        return get_default_value();
    }

    virtual bool is_synthetic_param() const {
        return false;
    }

    EXPORT void fail_wrong_type(const char *type);

private:
    explicit GeneratorParamBase(const GeneratorParamBase &) = delete;
    void operator=(const GeneratorParamBase &) = delete;

    // Generator which owns this GeneratorParam. Note that this will be null
    // initially; the GeneratorBase itself will set this field when it initially
    // builds its info about params. However, since it (generally) isn't
    // appropriate for GeneratorParam<> to be declared outside of a Generator,
    // all reasonable non-testing code should expect this to be non-null.
    GeneratorBase *generator{nullptr};
};

template<typename T>
class GeneratorParamImpl : public GeneratorParamBase {
public:
    using type = T;

    GeneratorParamImpl(const std::string &name, const T &value) : GeneratorParamBase(name), value_(value) {}

    T value() const { check_value_readable(); return value_; }

    operator T() const { return this->value(); }

    operator Expr() const { return make_const(type_of<T>(), this->value()); }

#define HALIDE_GENERATOR_PARAM_TYPED_SETTER(TYPE) \
    void set(const TYPE &new_value) override { typed_setter_impl<TYPE>(new_value, #TYPE); }

    HALIDE_GENERATOR_PARAM_TYPED_SETTER(bool)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int8_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int16_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int32_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(int64_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint8_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint16_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint32_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(uint64_t)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(float)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(double)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(Target)
    HALIDE_GENERATOR_PARAM_TYPED_SETTER(Type)

#undef HALIDE_GENERATOR_PARAM_TYPED_SETTER

protected:
    virtual void set_impl(const T &new_value) { check_value_writable(); value_ = new_value; }

private:
    T value_;

    template <typename T2, typename std::enable_if<std::is_convertible<T2, T>::value>::type * = nullptr>
    HALIDE_ALWAYS_INLINE void typed_setter_impl(const T2 &value, const char * msg) {
        check_value_writable();
        // Arithmetic types must roundtrip losslessly.
        if (!std::is_same<T, T2>::value &&
            std::is_arithmetic<T>::value &&
            std::is_arithmetic<T2>::value) {
            const T t = Convert<T2, T>::value(value);
            const T2 t2 = Convert<T, T2>::value(t);
            if (t2 != value) {
                fail_wrong_type(msg);
            }
        }
        value_ = Convert<T2, T>::value(value);
    }

    template <typename T2, typename std::enable_if<!std::is_convertible<T2, T>::value>::type * = nullptr>
    HALIDE_ALWAYS_INLINE void typed_setter_impl(const T2 &, const char *msg) {
        fail_wrong_type(msg);
    }
};

// Stubs for type-specific implementations of GeneratorParam, to avoid
// many complex enable_if<> statements that were formerly spread through the
// implementation. Note that not all of these need to be templated classes,
// (e.g. for GeneratorParam_Target, T == Target always), but are declared
// that way for symmetry of declaration.
template<typename T>
class GeneratorParam_Target : public GeneratorParamImpl<T> {
public:
    GeneratorParam_Target(const std::string &name, const T &value) : GeneratorParamImpl<T>(name, value) {}

    void set_from_string(const std::string &new_value_string) override {
        this->set(Target(new_value_string));
    }

    std::string to_string() const override {
        return this->value().to_string();
    }

    std::string call_to_string(const std::string &v) const override {
        std::ostringstream oss;
        oss << v << ".to_string()";
        return oss.str();
    }

    std::string get_c_type() const override {
        return "Target";
    }
};

template<typename T>
class GeneratorParam_Arithmetic : public GeneratorParamImpl<T> {
public:
    GeneratorParam_Arithmetic(const std::string &name,
                              const T &value,
                              const T &min = std::numeric_limits<T>::lowest(),
                              const T &max = std::numeric_limits<T>::max())
        : GeneratorParamImpl<T>(name, value), min(min), max(max) {
        // call set() to ensure value is clamped to min/max
        this->set(value);
    }

    void set_impl(const T &new_value) override {
        user_assert(new_value >= min && new_value <= max) << "Value out of range: " << new_value;
        GeneratorParamImpl<T>::set_impl(new_value);
    }

    void set_from_string(const std::string &new_value_string) override {
        std::istringstream iss(new_value_string);
        T t;
        iss >> t;
        user_assert(!iss.fail() && iss.get() == EOF) << "Unable to parse: " << new_value_string;
        this->set(t);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << this->value();
        if (std::is_same<T, float>::value) {
            // If the constant has no decimal point ("1")
            // we must append one before appending "f"
            if (oss.str().find(".") == std::string::npos) {
                oss << ".";
            }
            oss << "f";
        }
        return oss.str();
    }

    std::string call_to_string(const std::string &v) const override {
        std::ostringstream oss;
        oss << "std::to_string(" << v << ")";
        return oss.str();
    }

    std::string get_c_type() const override {
        std::ostringstream oss;
        if (std::is_same<T, float>::value) {
            return "float";
        } else if (std::is_same<T, double>::value) {
            return "double";
        } else if (std::is_integral<T>::value) {
            if (std::is_unsigned<T>::value) oss << 'u';
            oss << "int" << (sizeof(T) * 8) << "_t";
            return oss.str();
        } else {
            user_error << "Unknown arithmetic type\n";
            return "";
        }
    }

private:
    const T min, max;
};

template<typename T>
class GeneratorParam_Bool : public GeneratorParam_Arithmetic<T> {
public:
    GeneratorParam_Bool(const std::string &name, const T &value) : GeneratorParam_Arithmetic<T>(name, value) {}

    void set_from_string(const std::string &new_value_string) override {
        bool v = false;
        if (new_value_string == "true") {
            v = true;
        } else if (new_value_string == "false") {
            v = false;
        } else {
            user_assert(false) << "Unable to parse bool: " << new_value_string;
        }
        this->set(v);
    }

    std::string to_string() const override {
        return this->value() ? "true" : "false";
    }

    std::string call_to_string(const std::string &v) const override {
        std::ostringstream oss;
        oss << "(" << v << ") ? \"true\" : \"false\"";
        return oss.str();
    }

    std::string get_c_type() const override {
        return "bool";
    }
};

template<typename T>
class GeneratorParam_Enum : public GeneratorParamImpl<T> {
public:
    GeneratorParam_Enum(const std::string &name, const T &value, const std::map<std::string, T> &enum_map)
        : GeneratorParamImpl<T>(name, value), enum_map(enum_map) {}

    // define a "set" that takes our specific enum (but don't hide the inherited virtual functions)
    using GeneratorParamImpl<T>::set;

    template <typename T2 = T, typename std::enable_if<!std::is_same<T2, Type>::value>::type * = nullptr>
    void set(const T &e) {
        this->set_impl(e);
    }

    void set_from_string(const std::string &new_value_string) override {
        auto it = enum_map.find(new_value_string);
        user_assert(it != enum_map.end()) << "Enumeration value not found: " << new_value_string;
        this->set_impl(it->second);
    }

    std::string to_string() const override {
        return enum_to_string(enum_map, this->value());
    }

    std::string call_to_string(const std::string &v) const override {
        return "Enum_" + this->name + "_map().at(" + v + ")";
    }

    std::string get_c_type() const override {
        return "Enum_" + this->name;
    }

    std::string get_default_value() const override {
        return "Enum_" + this->name + "::" + enum_to_string(enum_map, this->value());
    }

    std::string get_type_decls() const override {
        std::ostringstream oss;
        oss << "enum class Enum_" << this->name << " {\n";
        for (auto key_value : enum_map) {
            oss << "  " << key_value.first << ",\n";
        }
        oss << "};\n";
        oss << "\n";

        // TODO: since we generate the enums, we could probably just use a vector (or array!) rather than a map,
        // since we can ensure that the enum values are a nice tight range.
        oss << "inline NO_INLINE const std::map<Enum_" << this->name << ", std::string>& Enum_" << this->name << "_map() {\n";
        oss << "  static const std::map<Enum_" << this->name << ", std::string> m = {\n";
        for (auto key_value : enum_map) {
            oss << "    { Enum_" << this->name << "::" << key_value.first << ", \"" << key_value.first << "\"},\n";
        }
        oss << "  };\n";
        oss << "  return m;\n";
        oss << "};\n";
        return oss.str();
    }

private:
    const std::map<std::string, T> enum_map;
};

template<typename T>
class GeneratorParam_Type : public GeneratorParam_Enum<T> {
public:
    GeneratorParam_Type(const std::string &name, const T &value)
        : GeneratorParam_Enum<T>(name, value, get_halide_type_enum_map()) {}

    std::string call_to_string(const std::string &v) const override {
        return "Halide::Internal::halide_type_to_enum_string(" + v + ")";
    }

    std::string get_c_type() const override {
        return "Type";
    }

    std::string get_template_type() const override {
        return "typename";
    }

    std::string get_template_value() const override {
        return halide_type_to_c_type(this->value());
    }

    std::string get_default_value() const override {
        return halide_type_to_c_source(this->value());
    }

    std::string get_type_decls() const override {
        return "";
    }
};

template<typename T>
using GeneratorParamImplBase =
    typename select_type<
        cond<std::is_same<T, Target>::value,    GeneratorParam_Target<T>>,
        cond<std::is_same<T, Type>::value,      GeneratorParam_Type<T>>,
        cond<std::is_same<T, bool>::value,      GeneratorParam_Bool<T>>,
        cond<std::is_arithmetic<T>::value,      GeneratorParam_Arithmetic<T>>,
        cond<std::is_enum<T>::value,            GeneratorParam_Enum<T>>
    >::type;

}  // namespace Internal

/** GeneratorParam is a templated class that can be used to modify the behavior
 * of the Generator at code-generation time. GeneratorParams are commonly
 * specified in build files (e.g. Makefile) to customize the behavior of
 * a given Generator, thus they have a very constrained set of types to allow
 * for efficient specification via command-line flags. A GeneratorParam can be:
 *   - any float or int type.
 *   - bool
 *   - enum
 *   - Halide::Target
 *   - Halide::Type
 * All GeneratorParams have a default value. Arithmetic types can also
 * optionally specify min and max. Enum types must specify a string-to-value
 * map.
 *
 * Halide::Type is treated as though it were an enum, with the mappings:
 *
 *   "int8"     Halide::Int(8)
 *   "int16"    Halide::Int(16)
 *   "int32"    Halide::Int(32)
 *   "uint8"    Halide::UInt(8)
 *   "uint16"   Halide::UInt(16)
 *   "uint32"   Halide::UInt(32)
 *   "float32"  Halide::Float(32)
 *   "float64"  Halide::Float(64)
 *
 * No vector Types are currently supported by this mapping.
 *
 */
template <typename T>
class GeneratorParam : public Internal::GeneratorParamImplBase<T> {
public:
    GeneratorParam(const std::string &name, const T &value)
        : Internal::GeneratorParamImplBase<T>(name, value) {}

    GeneratorParam(const std::string &name, const T &value, const T &min, const T &max)
        : Internal::GeneratorParamImplBase<T>(name, value, min, max) {}

    GeneratorParam(const std::string &name, const T &value, const std::map<std::string, T> &enum_map)
        : Internal::GeneratorParamImplBase<T>(name, value, enum_map) {}

    GeneratorParam(const std::string &name, const std::string &value)
        : Internal::GeneratorParamImplBase<T>(name, value) {}
};


/** Addition between GeneratorParam<T> and any type that supports operator+ with T.
 * Returns type of underlying operator+. */
// @{
template <typename Other, typename T>
decltype((Other)0 + (T)0) operator+(const Other &a, const GeneratorParam<T> &b) { return a + (T)b; }
template <typename Other, typename T>
decltype((T)0 + (Other)0) operator+(const GeneratorParam<T> &a, const Other & b) { return (T)a + b; }
// @}

/** Subtraction between GeneratorParam<T> and any type that supports operator- with T.
 * Returns type of underlying operator-. */
// @{
template <typename Other, typename T>
decltype((Other)0 - (T)0) operator-(const Other & a, const GeneratorParam<T> &b) { return a - (T)b; }
template <typename Other, typename T>
decltype((T)0 - (Other)0)  operator-(const GeneratorParam<T> &a, const Other & b) { return (T)a - b; }
// @}

/** Multiplication between GeneratorParam<T> and any type that supports operator* with T.
 * Returns type of underlying operator*. */
// @{
template <typename Other, typename T>
decltype((Other)0 * (T)0) operator*(const Other &a, const GeneratorParam<T> &b) { return a * (T)b; }
template <typename Other, typename T>
decltype((Other)0 * (T)0) operator*(const GeneratorParam<T> &a, const Other &b) { return (T)a * b; }
// @}

/** Division between GeneratorParam<T> and any type that supports operator/ with T.
 * Returns type of underlying operator/. */
// @{
template <typename Other, typename T>
decltype((Other)0 / (T)1) operator/(const Other &a, const GeneratorParam<T> &b) { return a / (T)b; }
template <typename Other, typename T>
decltype((T)0 / (Other)1) operator/(const GeneratorParam<T> &a, const Other &b) { return (T)a / b; }
// @}

/** Modulo between GeneratorParam<T> and any type that supports operator% with T.
 * Returns type of underlying operator%. */
// @{
template <typename Other, typename T>
decltype((Other)0 % (T)1) operator%(const Other &a, const GeneratorParam<T> &b) { return a % (T)b; }
template <typename Other, typename T>
decltype((T)0 % (Other)1) operator%(const GeneratorParam<T> &a, const Other &b) { return (T)a % b; }
// @}

/** Greater than comparison between GeneratorParam<T> and any type that supports operator> with T.
 * Returns type of underlying operator>. */
// @{
template <typename Other, typename T>
decltype((Other)0 > (T)1) operator>(const Other &a, const GeneratorParam<T> &b) { return a > (T)b; }
template <typename Other, typename T>
decltype((T)0 > (Other)1) operator>(const GeneratorParam<T> &a, const Other &b) { return (T)a > b; }
// @}

/** Less than comparison between GeneratorParam<T> and any type that supports operator< with T.
 * Returns type of underlying operator<. */
// @{
template <typename Other, typename T>
decltype((Other)0 < (T)1) operator<(const Other &a, const GeneratorParam<T> &b) { return a < (T)b; }
template <typename Other, typename T>
decltype((T)0 < (Other)1) operator<(const GeneratorParam<T> &a, const Other &b) { return (T)a < b; }
// @}

/** Greater than or equal comparison between GeneratorParam<T> and any type that supports operator>= with T.
 * Returns type of underlying operator>=. */
// @{
template <typename Other, typename T>
decltype((Other)0 >= (T)1) operator>=(const Other &a, const GeneratorParam<T> &b) { return a >= (T)b; }
template <typename Other, typename T>
decltype((T)0 >= (Other)1) operator>=(const GeneratorParam<T> &a, const Other &b) { return (T)a >= b; }
// @}

/** Less than or equal comparison between GeneratorParam<T> and any type that supports operator<= with T.
 * Returns type of underlying operator<=. */
// @{
template <typename Other, typename T>
decltype((Other)0 <= (T)1) operator<=(const Other &a, const GeneratorParam<T> &b) { return a <= (T)b; }
template <typename Other, typename T>
decltype((T)0 <= (Other)1) operator<=(const GeneratorParam<T> &a, const Other &b) { return (T)a <= b; }
// @}

/** Equality comparison between GeneratorParam<T> and any type that supports operator== with T.
 * Returns type of underlying operator==. */
// @{
template <typename Other, typename T>
decltype((Other)0 == (T)1) operator==(const Other &a, const GeneratorParam<T> &b) { return a == (T)b; }
template <typename Other, typename T>
decltype((T)0 == (Other)1) operator==(const GeneratorParam<T> &a, const Other &b) { return (T)a == b; }
// @}

/** Inequality comparison between between GeneratorParam<T> and any type that supports operator!= with T.
 * Returns type of underlying operator!=. */
// @{
template <typename Other, typename T>
decltype((Other)0 != (T)1) operator!=(const Other &a, const GeneratorParam<T> &b) { return a != (T)b; }
template <typename Other, typename T>
decltype((T)0 != (Other)1) operator!=(const GeneratorParam<T> &a, const Other &b) { return (T)a != b; }
// @}

/** Logical and between between GeneratorParam<T> and any type that supports operator&& with T.
 * Returns type of underlying operator&&. */
// @{
template <typename Other, typename T>
decltype((Other)0 && (T)1) operator&&(const Other &a, const GeneratorParam<T> &b) { return a && (T)b; }
template <typename Other, typename T>
decltype((T)0 && (Other)1) operator&&(const GeneratorParam<T> &a, const Other &b) { return (T)a && b; }
// @}

/** Logical or between between GeneratorParam<T> and any type that supports operator&& with T.
 * Returns type of underlying operator||. */
// @{
template <typename Other, typename T>
decltype((Other)0 || (T)1) operator||(const Other &a, const GeneratorParam<T> &b) { return a || (T)b; }
template <typename Other, typename T>
decltype((T)0 || (Other)1) operator||(const GeneratorParam<T> &a, const Other &b) { return (T)a || b; }
// @}

/* min and max are tricky as the language support for these is in the std
 * namespace. In order to make this work, forwarding functions are used that
 * are declared in a namespace that has std::min and std::max in scope.
 */
namespace Internal { namespace GeneratorMinMax {

using std::max;
using std::min;

template <typename Other, typename T>
decltype(min((Other)0, (T)1)) min_forward(const Other &a, const GeneratorParam<T> &b) { return min(a, (T)b); }
template <typename Other, typename T>
decltype(min((T)0, (Other)1)) min_forward(const GeneratorParam<T> &a, const Other &b) { return min((T)a, b); }

template <typename Other, typename T>
decltype(max((Other)0, (T)1)) max_forward(const Other &a, const GeneratorParam<T> &b) { return max(a, (T)b); }
template <typename Other, typename T>
decltype(max((T)0, (Other)1)) max_forward(const GeneratorParam<T> &a, const Other &b) { return max((T)a, b); }

}}

/** Compute minimum between GeneratorParam<T> and any type that supports min with T.
 * Will automatically import std::min. Returns type of underlying min call. */
// @{
template <typename Other, typename T>
auto min(const Other &a, const GeneratorParam<T> &b) -> decltype(Internal::GeneratorMinMax::min_forward(a, b)) {
    return Internal::GeneratorMinMax::min_forward(a, b);
}
template <typename Other, typename T>
auto min(const GeneratorParam<T> &a, const Other &b) -> decltype(Internal::GeneratorMinMax::min_forward(a, b)) {
    return Internal::GeneratorMinMax::min_forward(a, b);
}
// @}

/** Compute the maximum value between GeneratorParam<T> and any type that supports max with T.
 * Will automatically import std::max. Returns type of underlying max call. */
// @{
template <typename Other, typename T>
auto max(const Other &a, const GeneratorParam<T> &b) -> decltype(Internal::GeneratorMinMax::max_forward(a, b)) {
    return Internal::GeneratorMinMax::max_forward(a, b);
}
template <typename Other, typename T>
auto max(const GeneratorParam<T> &a, const Other &b) -> decltype(Internal::GeneratorMinMax::max_forward(a, b)) {
    return Internal::GeneratorMinMax::max_forward(a, b);
}
// @}

/** Not operator for GeneratorParam */
template <typename T>
decltype(!(T)0) operator!(const GeneratorParam<T> &a) { return !(T)a; }

namespace Internal {

template<typename T2> class GeneratorInput_Buffer;

enum class IOKind { Scalar, Function, Buffer };

/**
 * StubInputBuffer is the placeholder that a Stub uses when it requires
 * a Buffer for an input (rather than merely a Func or Expr). It is constructed
 * to allow only two possible sorts of input:
 * -- Assignment of an Input<Buffer<>>, with compatible type and dimensions,
 * essentially allowing us to pipe a parameter from an enclosing Generator to an internal Stub.
 * -- Assignment of a Buffer<>, with compatible type and dimensions,
 * causing the Input<Buffer<>> to become a precompiled buffer in the generated code.
 */
template<typename T = void>
class StubInputBuffer {
    friend class StubInput;
    template<typename T2> friend class GeneratorInput_Buffer;

    Parameter parameter_;

    NO_INLINE explicit StubInputBuffer(const Parameter &p) : parameter_(p) {
        // Create an empty 1-element buffer with the right runtime typing and dimensions,
        // which we'll use only to pass to can_convert_from() to verify this
        // Parameter is compatible with our constraints.
        Buffer<> other(p.type(), nullptr, std::vector<int>(p.dimensions(), 1));
        internal_assert((Buffer<T>::can_convert_from(other)));
    }

    template<typename T2>
    NO_INLINE static Parameter parameter_from_buffer(const Buffer<T2> &b) {
        user_assert((Buffer<T>::can_convert_from(b)));
        Parameter p(b.type(), true, b.dimensions());
        p.set_buffer(b);
        return p;
    }

public:
    StubInputBuffer() {}

    // *not* explicit -- this ctor should only be used when you want
    // to pass a literal Buffer<> for a Stub Input; this Buffer<> will be
    // compiled into the Generator's product, rather than becoming
    // a runtime Parameter.
    template<typename T2>
    StubInputBuffer(const Buffer<T2> &b) : parameter_(parameter_from_buffer(b)) {}
};

class StubOutputBufferBase {
protected:
    Func f;
    std::shared_ptr<GeneratorBase> generator;

    EXPORT void check_scheduled(const char* m) const;
    EXPORT Target get_target() const;

    explicit StubOutputBufferBase(const Func &f, std::shared_ptr<GeneratorBase> generator) : f(f), generator(generator) {}
    StubOutputBufferBase() {}

public:
    Realization realize(std::vector<int32_t> sizes) {
        check_scheduled("realize");
        return f.realize(sizes, get_target());
    }

    template <typename... Args>
    Realization realize(Args&&... args) {
        check_scheduled("realize");
        return f.realize(std::forward<Args>(args)..., get_target());
    }

    template<typename Dst>
    void realize(Dst dst) {
        check_scheduled("realize");
        f.realize(dst, get_target());
    }
};

/**
 * StubOutputBuffer is the placeholder that a Stub uses when it requires
 * a Buffer for an output (rather than merely a Func). It is constructed
 * to allow only two possible sorts of things:
 * -- Assignment to an Output<Buffer<>>, with compatible type and dimensions,
 * essentially allowing us to pipe a parameter from the result of a Stub to an
 * enclosing Generator
 * -- Realization into a Buffer<>; this is useful only in JIT compilation modes
 * (and shouldn't be usable otherwise)
 *
 * It is deliberate that StubOutputBuffer is not (easily) convertible to Func.
 */
template<typename T = void>
class StubOutputBuffer : public StubOutputBufferBase {
    template<typename T2> friend class GeneratorOutput_Buffer;
    friend class GeneratorStub;
    explicit StubOutputBuffer(const Func &f, std::shared_ptr<GeneratorBase> generator) : StubOutputBufferBase(f, generator) {}
public:
    StubOutputBuffer() {}
};

// This is a union-like class that allows for convenient initialization of Stub Inputs
// via C++11 initializer-list syntax; it is only used in situations where the
// downstream consumer will be able to explicitly check that each value is
// of the expected/required kind.
class StubInput {
    const IOKind kind_;
    // Exactly one of the following fields should be defined:
    const Parameter parameter_;
    const Func func_;
    const Expr expr_;
public:
    // *not* explicit.
    template<typename T2>
    StubInput(const StubInputBuffer<T2> &b) : kind_(IOKind::Buffer), parameter_(b.parameter_) {}
    StubInput(const Func &f) : kind_(IOKind::Function), func_(f) {}
    StubInput(const Expr &e) : kind_(IOKind::Scalar), expr_(e) {}

private:
    friend class GeneratorInputBase;

    IOKind kind() const {
        return kind_;
    }

    Parameter parameter() const {
        internal_assert(kind_ == IOKind::Buffer);
        return parameter_;
    }

    Func func() const {
        internal_assert(kind_ == IOKind::Function);
        return func_;
    }

    Expr expr() const {
        internal_assert(kind_ == IOKind::Scalar);
        return expr_;
    }
};

class Constrainable {
public:
    virtual ~Constrainable() {}

    virtual Parameter parameter() const = 0;

    int dimensions() const {
        return parameter().dimensions();
    }

    Dimension dim(int i) {
        return Dimension(parameter(), i);
    }

    const Dimension dim(int i) const {
        return Dimension(parameter(), i);
    }

    int host_alignment() const {
        return parameter().host_alignment();
    }

    Constrainable &set_host_alignment(int alignment) {
        parameter().set_host_alignment(alignment);
        return *this;
    }

    const Expr left() const { return dim(0).min(); }
    const Expr right() const { return dim(0).max(); }
    const Expr top() const { return dim(1).min(); }
    const Expr bottom() const { return dim(1).max(); }

    const Expr width() const { return dim(0).extent(); }
    const Expr height() const { return dim(1).extent(); }
    const Expr channels() const { return dim(2).extent(); }
};

/** GIOBase is the base class for all GeneratorInput<> and GeneratorOutput<>
 * instantiations; it is not part of the public API and should never be
 * used directly by user code.
 *
 * Every GIOBase instance can be either a single value or an array-of-values;
 * each of these values can be an Expr or a Func. (Note that for an
 * array-of-values, the types/dimensions of all values in the array must match.)
 *
 * A GIOBase can have multiple Types, in which case it represents a Tuple.
 * (Note that Tuples are currently only supported for GeneratorOutput, but
 * it is likely that GeneratorInput will be extended to support Tuple as well.)
 *
 * The array-size, type(s), and dimensions can all be left "unspecified" at
 * creation time, in which case they may assume values provided by a Stub.
 * (It is important to note that attempting to use a GIOBase with unspecified
 * values will assert-fail; you must ensure that all unspecified values are
 * filled in prior to use.)
 */
class GIOBase {
public:
    EXPORT bool array_size_defined() const;
    EXPORT size_t array_size() const;
    EXPORT virtual bool is_array() const;

    EXPORT const std::string &name() const;
    EXPORT IOKind kind() const;

    EXPORT bool types_defined() const;
    EXPORT const std::vector<Type> &types() const;
    EXPORT Type type() const;

    EXPORT bool dims_defined() const;
    EXPORT int dims() const;

    EXPORT const std::vector<Func> &funcs() const;
    EXPORT const std::vector<Expr> &exprs() const;

protected:
    EXPORT GIOBase(size_t array_size,
                   const std::string &name,
                   IOKind kind,
                   const std::vector<Type> &types,
                   int dims);
    EXPORT virtual ~GIOBase();

    friend class GeneratorBase;

    int array_size_;           // always 1 if is_array() == false.
                               // -1 if is_array() == true but unspecified.

    const std::string name_;
    const IOKind kind_;
    std::vector<Type> types_;  // empty if type is unspecified
    int dims_;           // -1 if dim is unspecified

    // Exactly one of these will have nonzero length
    std::vector<Func> funcs_;
    std::vector<Expr> exprs_;

    // Generator which owns this Input or Output. Note that this will be null
    // initially; the GeneratorBase itself will set this field when it initially
    // builds its info about params. However, since it isn't
    // appropriate for Input<> or Output<> to be declared outside of a Generator,
    // all reasonable non-testing code should expect this to be non-null.
    GeneratorBase *generator{nullptr};

    EXPORT std::string array_name(size_t i) const;

    EXPORT virtual void verify_internals() const;

    EXPORT void check_matching_array_size(size_t size);
    EXPORT void check_matching_type_and_dim(const std::vector<Type> &t, int d);

    template<typename ElemType>
    const std::vector<ElemType> &get_values() const;

    virtual bool allow_synthetic_generator_params() const {
        return true;
    }

    virtual Parameter parameter() const {
        internal_error << "Unimplemented";
        return Parameter();
    }

    virtual void check_value_writable() const = 0;

private:
    template<typename T> friend class GeneratorParam_Synthetic;

    explicit GIOBase(const GIOBase &) = delete;
    void operator=(const GIOBase &) = delete;
};

template<>
inline const std::vector<Expr> &GIOBase::get_values<Expr>() const {
    return exprs();
}

template<>
inline const std::vector<Func> &GIOBase::get_values<Func>() const {
    return funcs();
}

class GeneratorInputBase : public GIOBase {
protected:
    EXPORT GeneratorInputBase(size_t array_size,
                       const std::string &name,
                       IOKind kind,
                       const std::vector<Type> &t,
                       int d);

    EXPORT GeneratorInputBase(const std::string &name, IOKind kind, const std::vector<Type> &t, int d);

    EXPORT ~GeneratorInputBase() override;

    friend class GeneratorBase;

    std::vector<Parameter> parameters_;

    EXPORT void init_internals();
    EXPORT void set_inputs(const std::vector<StubInput> &inputs);

    EXPORT virtual void set_def_min_max();

    EXPORT void verify_internals() const override;

    friend class StubEmitter;

    virtual std::string get_c_type() const = 0;

    EXPORT void check_value_writable() const override;

    EXPORT void estimate_impl(Var var, Expr min, Expr extent);

private:
    EXPORT void init_parameters();
};


template<typename T, typename ValueType>
class GeneratorInputImpl : public GeneratorInputBase {
protected:
    using TBase = typename std::remove_all_extents<T>::type;

    bool is_array() const override {
        return std::is_array<T>::value;
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2 not-an-array
        !std::is_array<T2>::value
    >::type * = nullptr>
    GeneratorInputImpl(const std::string &name, IOKind kind, const std::vector<Type> &t, int d)
        : GeneratorInputBase(name, kind, t, d) {
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2[kSomeConst]
        std::is_array<T2>::value && std::rank<T2>::value == 1 && (std::extent<T2, 0>::value > 0)
    >::type * = nullptr>
    GeneratorInputImpl(const std::string &name, IOKind kind, const std::vector<Type> &t, int d)
        : GeneratorInputBase(std::extent<T2, 0>::value, name, kind, t, d) {
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2[]
        std::is_array<T2>::value && std::rank<T2>::value == 1 && std::extent<T2, 0>::value == 0
    >::type * = nullptr>
    GeneratorInputImpl(const std::string &name, IOKind kind, const std::vector<Type> &t, int d)
        : GeneratorInputBase(-1, name, kind, t, d) {
    }

public:
    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    size_t size() const {
        return get_values<ValueType>().size();
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    const ValueType &operator[](size_t i) const {
        return get_values<ValueType>()[i];
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    const ValueType &at(size_t i) const {
        return get_values<ValueType>().at(i);
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    typename std::vector<ValueType>::const_iterator begin() const {
        return get_values<ValueType>().begin();
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    typename std::vector<ValueType>::const_iterator end() const {
        return get_values<ValueType>().end();
    }
};

template<typename T>
class GeneratorInput_Buffer : public GeneratorInputImpl<T, Func>, public Constrainable {
private:
    using Super = GeneratorInputImpl<T, Func>;

protected:
    using TBase = typename Super::TBase;

    friend class ::Halide::Func;
    friend class ::Halide::Stage;

    bool allow_synthetic_generator_params() const override {
        return !T::has_static_halide_type;
    }

    std::string get_c_type() const override {
        if (T::has_static_halide_type) {
            return "Halide::Internal::StubInputBuffer<" +
                halide_type_to_c_type(T::static_halide_type()) +
                ">";
        } else {
            return "Halide::Internal::StubInputBuffer<>";
        }
    }

    Parameter parameter() const override {
        internal_assert(this->parameters_.size() == 1);
        return this->parameters_.at(0);
    }

public:
    GeneratorInput_Buffer(const std::string &name)
        : Super(name, IOKind::Buffer,
                T::has_static_halide_type ? std::vector<Type>{ T::static_halide_type() } : std::vector<Type>{},
                -1) {
    }

    GeneratorInput_Buffer(const std::string &name, const Type &t, int d = -1)
        : Super(name, IOKind::Buffer, {t}, d) {
        static_assert(!T::has_static_halide_type, "Cannot use pass a Type argument for a Buffer with a non-void static type");
    }

    GeneratorInput_Buffer(const std::string &name, int d)
        : Super(name, IOKind::Buffer, T::has_static_halide_type ? std::vector<Type>{ T::static_halide_type() } : std::vector<Type>{}, d) {
    }


    template <typename... Args>
    Expr operator()(Args&&... args) const {
        return this->funcs().at(0)(std::forward<Args>(args)...);
    }

    Expr operator()(std::vector<Expr> args) const {
        return this->funcs().at(0)(args);
    }

    template<typename T2>
    operator StubInputBuffer<T2>() const {
        return StubInputBuffer<T2>(parameter());
    }

    operator Func() const {
        return this->funcs().at(0);
    }

    operator ExternFuncArgument() const {
        return ExternFuncArgument(this->parameters_.at(0));
    }

    GeneratorInput_Buffer<T> &estimate(Var var, Expr min, Expr extent) {
        this->estimate_impl(var, min, extent);
        return *this;
    }

    Func in() {
        return Func(*this).in();
    }

    Func in(Func other) {
        return Func(*this).in(other);
    }

    Func in(const std::vector<Func> &others) {
        return Func(*this).in(others);
    }
};


template<typename T>
class GeneratorInput_Func : public GeneratorInputImpl<T, Func> {
private:
    using Super = GeneratorInputImpl<T, Func>;

protected:
    using TBase = typename Super::TBase;

    std::string get_c_type() const override {
        return "Func";
    }

public:
    GeneratorInput_Func(const std::string &name, const Type &t, int d)
        : Super(name, IOKind::Function, {t}, d) {
    }

    // unspecified type
    GeneratorInput_Func(const std::string &name, int d)
        : Super(name, IOKind::Function, {}, d) {
    }

    // unspecified dimension
    GeneratorInput_Func(const std::string &name, const Type &t)
        : Super(name, IOKind::Function, {t}, -1) {
    }

    // unspecified type & dimension
    GeneratorInput_Func(const std::string &name)
        : Super(name, IOKind::Function, {}, -1) {
    }

    GeneratorInput_Func(size_t array_size, const std::string &name, const Type &t, int d)
        : Super(array_size, name, IOKind::Function, {t}, d) {
    }

    // unspecified type
    GeneratorInput_Func(size_t array_size, const std::string &name, int d)
        : Super(array_size, name, IOKind::Function, {}, d) {
    }

    // unspecified dimension
    GeneratorInput_Func(size_t array_size, const std::string &name, const Type &t)
        : Super(array_size, name, IOKind::Function, {t}, -1) {
    }

    // unspecified type & dimension
    GeneratorInput_Func(size_t array_size, const std::string &name)
        : Super(array_size, name, IOKind::Function, {}, -1) {
    }

    template <typename... Args>
    Expr operator()(Args&&... args) const {
        return this->funcs().at(0)(std::forward<Args>(args)...);
    }

    Expr operator()(std::vector<Expr> args) const {
        return this->funcs().at(0)(args);
    }

    operator Func() const {
        return this->funcs().at(0);
    }

    operator ExternFuncArgument() const {
        return ExternFuncArgument(this->parameters_.at(0));
    }

    GeneratorInput_Func<T> &estimate(Var var, Expr min, Expr extent) {
        this->estimate_impl(var, min, extent);
        return *this;
    }

    Func in() {
        return Func(*this).in();
    }

    Func in(Func other) {
        return Func(*this).in(other);
    }

    Func in(const std::vector<Func> &others) {
        return Func(*this).in(others);
    }
};


template<typename T>
class GeneratorInput_Scalar : public GeneratorInputImpl<T, Expr> {
private:
    using Super = GeneratorInputImpl<T, Expr>;
protected:
    using TBase = typename Super::TBase;

    const TBase def_{TBase()};

protected:
    void set_def_min_max() override {
        for (Parameter &p : this->parameters_) {
            p.set_scalar<TBase>(def_);
        }
    }

    std::string get_c_type() const override {
        return "Expr";
    }

public:
    explicit GeneratorInput_Scalar(const std::string &name,
                                   const TBase &def = static_cast<TBase>(0))
        : Super(name, IOKind::Scalar, {type_of<TBase>()}, 0), def_(def) {
    }

    GeneratorInput_Scalar(size_t array_size,
                          const std::string &name,
                          const TBase &def = static_cast<TBase>(0))
        : Super(array_size, name, IOKind::Scalar, {type_of<TBase>()}, 0), def_(def) {
    }

    /** You can use this Input as an expression in a halide
     * function definition */
    operator Expr() const {
        return this->exprs().at(0);
    }

    /** Using an Input as the argument to an external stage treats it
     * as an Expr */
    operator ExternFuncArgument() const {
        return ExternFuncArgument(this->exprs().at(0));
    }

    void set_estimate(const T &value) {
        for (Parameter &p : this->parameters_) {
            p.set_estimate(Expr(value));
        }
    }
};

template<typename T>
class GeneratorInput_Arithmetic : public GeneratorInput_Scalar<T> {
private:
    using Super = GeneratorInput_Scalar<T>;
protected:
    using TBase = typename Super::TBase;

    const Expr min_, max_;

protected:
    void set_def_min_max() override {
        GeneratorInput_Scalar<T>::set_def_min_max();
        // Don't set min/max for bool
        if (!std::is_same<TBase, bool>::value) {
            for (Parameter &p : this->parameters_) {
                if (min_.defined()) p.set_min_value(min_);
                if (max_.defined()) p.set_max_value(max_);
            }
        }
    }

public:
    explicit GeneratorInput_Arithmetic(const std::string &name,
                                       const TBase &def = static_cast<TBase>(0))
        : Super(name, def), min_(Expr()), max_(Expr()) {
    }

    GeneratorInput_Arithmetic(size_t array_size,
                              const std::string &name,
                              const TBase &def = static_cast<TBase>(0))
        : Super(array_size, name, def), min_(Expr()), max_(Expr()) {
    }

    GeneratorInput_Arithmetic(const std::string &name,
                              const TBase &def,
                              const TBase &min,
                              const TBase &max)
        : Super(name, def), min_(min), max_(max) {
    }

    GeneratorInput_Arithmetic(size_t array_size,
                              const std::string &name,
                              const TBase &def,
                              const TBase &min,
                              const TBase &max)
        : Super(array_size, name, def), min_(min), max_(max) {
    }
};

template<typename>
struct type_sink { typedef void type; };

template<typename T2, typename = void>
struct has_static_halide_type_method : std::false_type {};

template<typename T2>
struct has_static_halide_type_method<T2, typename type_sink<decltype(T2::static_halide_type())>::type> : std::true_type {};

template<typename T, typename TBase = typename std::remove_all_extents<T>::type>
using GeneratorInputImplBase =
    typename select_type<
        cond<has_static_halide_type_method<TBase>::value, GeneratorInput_Buffer<T>>,
        cond<std::is_same<TBase, Func>::value,            GeneratorInput_Func<T>>,
        cond<std::is_arithmetic<TBase>::value,            GeneratorInput_Arithmetic<T>>,
        cond<std::is_scalar<TBase>::value,                GeneratorInput_Scalar<T>>
    >::type;

}  // namespace Internal

template <typename T>
class GeneratorInput : public Internal::GeneratorInputImplBase<T> {
private:
    using Super = Internal::GeneratorInputImplBase<T>;
protected:
    using TBase = typename Super::TBase;

    // Trick to avoid ambiguous ctor between Func-with-dim and int-with-default-value;
    // since we can't use std::enable_if on ctors, define the argument to be one that
    // can only be properly resolved for TBase=Func.
    struct Unused;
    using IntIfNonScalar =
        typename Internal::select_type<
            Internal::cond<Internal::has_static_halide_type_method<TBase>::value, int>,
            Internal::cond<std::is_same<TBase, Func>::value, int>,
            Internal::cond<true, Unused>
        >::type;

public:
    explicit GeneratorInput(const std::string &name)
        : Super(name) {
    }

    GeneratorInput(const std::string &name, const TBase &def)
        : Super(name, def) {
    }

    GeneratorInput(size_t array_size, const std::string &name, const TBase &def)
        : Super(array_size, name, def) {
    }

    GeneratorInput(const std::string &name,
                   const TBase &def, const TBase &min, const TBase &max)
        : Super(name, def, min, max) {
    }

    GeneratorInput(size_t array_size, const std::string &name,
                   const TBase &def, const TBase &min, const TBase &max)
        : Super(array_size, name, def, min, max) {
    }

    GeneratorInput(const std::string &name, const Type &t, int d)
        : Super(name, t, d) {
    }

    GeneratorInput(const std::string &name, const Type &t)
        : Super(name, t) {
    }

    // Avoid ambiguity between Func-with-dim and int-with-default
    GeneratorInput(const std::string &name, IntIfNonScalar d)
        : Super(name, d) {
    }

    GeneratorInput(size_t array_size, const std::string &name, const Type &t, int d)
        : Super(array_size, name, t, d) {
    }

    GeneratorInput(size_t array_size, const std::string &name, const Type &t)
        : Super(array_size, name, t) {
    }

    // Avoid ambiguity between Func-with-dim and int-with-default
    //template <typename T2 = T, typename std::enable_if<std::is_same<TBase, Func>::value>::type * = nullptr>
    GeneratorInput(size_t array_size, const std::string &name, IntIfNonScalar d)
        : Super(array_size, name, d) {
    }

    GeneratorInput(size_t array_size, const std::string &name)
        : Super(array_size, name) {
    }
};

namespace Internal {


class GeneratorOutputBase : public GIOBase {
public:
#define HALIDE_OUTPUT_FORWARD(method)                                       \
    template<typename ...Args>                                              \
    inline auto method(Args&&... args) ->                                   \
        decltype(std::declval<Func>().method(std::forward<Args>(args)...)) {\
        return get_func_ref().method(std::forward<Args>(args)...);          \
    }

#define HALIDE_OUTPUT_FORWARD_CONST(method)                                 \
    template<typename ...Args>                                              \
    inline auto method(Args&&... args) const ->                             \
        decltype(std::declval<Func>().method(std::forward<Args>(args)...)) {\
        return get_func_ref().method(std::forward<Args>(args)...);          \
    }

    /** Forward schedule-related methods to the underlying Func. */
    // @{
    HALIDE_OUTPUT_FORWARD(align_bounds)
    HALIDE_OUTPUT_FORWARD(align_storage)
    HALIDE_OUTPUT_FORWARD_CONST(args)
    HALIDE_OUTPUT_FORWARD(bound)
    HALIDE_OUTPUT_FORWARD(bound_extent)
    HALIDE_OUTPUT_FORWARD(compute_at)
    HALIDE_OUTPUT_FORWARD(compute_inline)
    HALIDE_OUTPUT_FORWARD(compute_root)
    HALIDE_OUTPUT_FORWARD(define_extern)
    HALIDE_OUTPUT_FORWARD_CONST(defined)
    HALIDE_OUTPUT_FORWARD(fold_storage)
    HALIDE_OUTPUT_FORWARD(fuse)
    HALIDE_OUTPUT_FORWARD(glsl)
    HALIDE_OUTPUT_FORWARD(gpu)
    HALIDE_OUTPUT_FORWARD(gpu_blocks)
    HALIDE_OUTPUT_FORWARD(gpu_single_thread)
    HALIDE_OUTPUT_FORWARD(gpu_threads)
    HALIDE_OUTPUT_FORWARD(gpu_tile)
    HALIDE_OUTPUT_FORWARD_CONST(has_update_definition)
    HALIDE_OUTPUT_FORWARD(hexagon)
    HALIDE_OUTPUT_FORWARD(in)
    HALIDE_OUTPUT_FORWARD(memoize)
    HALIDE_OUTPUT_FORWARD_CONST(num_update_definitions)
    HALIDE_OUTPUT_FORWARD_CONST(output_types)
    HALIDE_OUTPUT_FORWARD_CONST(outputs)
    HALIDE_OUTPUT_FORWARD(parallel)
    HALIDE_OUTPUT_FORWARD(prefetch)
    HALIDE_OUTPUT_FORWARD(print_loop_nest)
    HALIDE_OUTPUT_FORWARD(rename)
    HALIDE_OUTPUT_FORWARD(reorder)
    HALIDE_OUTPUT_FORWARD(reorder_storage)
    HALIDE_OUTPUT_FORWARD_CONST(rvars)
    HALIDE_OUTPUT_FORWARD(serial)
    HALIDE_OUTPUT_FORWARD(shader)
    HALIDE_OUTPUT_FORWARD(specialize)
    HALIDE_OUTPUT_FORWARD(specialize_fail)
    HALIDE_OUTPUT_FORWARD(split)
    HALIDE_OUTPUT_FORWARD(store_at)
    HALIDE_OUTPUT_FORWARD(store_root)
    HALIDE_OUTPUT_FORWARD(tile)
    HALIDE_OUTPUT_FORWARD(trace_stores)
    HALIDE_OUTPUT_FORWARD(unroll)
    HALIDE_OUTPUT_FORWARD(update)
    HALIDE_OUTPUT_FORWARD_CONST(update_args)
    HALIDE_OUTPUT_FORWARD_CONST(update_value)
    HALIDE_OUTPUT_FORWARD_CONST(update_values)
    HALIDE_OUTPUT_FORWARD_CONST(value)
    HALIDE_OUTPUT_FORWARD_CONST(values)
    HALIDE_OUTPUT_FORWARD(vectorize)
    // }@

#undef HALIDE_OUTPUT_FORWARD

protected:
    EXPORT GeneratorOutputBase(size_t array_size,
                        const std::string &name,
                        IOKind kind,
                        const std::vector<Type> &t,
                        int d);

    EXPORT GeneratorOutputBase(const std::string &name,
                               IOKind kind,
                               const std::vector<Type> &t,
                               int d);

    EXPORT ~GeneratorOutputBase() override;

    friend class GeneratorBase;
    friend class StubEmitter;

    EXPORT void init_internals();
    EXPORT void resize(size_t size);

    virtual std::string get_c_type() const {
        return "Func";
    }

    EXPORT void check_value_writable() const override;

    NO_INLINE Func &get_func_ref() {
        internal_assert(kind() != IOKind::Scalar);
        internal_assert(funcs_.size() == array_size() && exprs_.empty());
        return funcs_[0];
    }

    NO_INLINE const Func &get_func_ref() const {
        internal_assert(kind() != IOKind::Scalar);
        internal_assert(funcs_.size() == array_size() && exprs_.empty());
        return funcs_[0];
    }
};

template<typename T>
class GeneratorOutputImpl : public GeneratorOutputBase {
protected:
    using TBase = typename std::remove_all_extents<T>::type;
    using ValueType = Func;

    bool is_array() const override {
        return std::is_array<T>::value;
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2 not-an-array
        !std::is_array<T2>::value
    >::type * = nullptr>
    GeneratorOutputImpl(const std::string &name, IOKind kind, const std::vector<Type> &t, int d)
        : GeneratorOutputBase(name, kind, t, d) {
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2[kSomeConst]
        std::is_array<T2>::value && std::rank<T2>::value == 1 && (std::extent<T2, 0>::value > 0)
    >::type * = nullptr>
    GeneratorOutputImpl(const std::string &name, IOKind kind, const std::vector<Type> &t, int d)
        : GeneratorOutputBase(std::extent<T2, 0>::value, name, kind, t, d) {
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2[]
        std::is_array<T2>::value && std::rank<T2>::value == 1 && std::extent<T2, 0>::value == 0
    >::type * = nullptr>
    GeneratorOutputImpl(const std::string &name, IOKind kind, const std::vector<Type> &t, int d)
        : GeneratorOutputBase(-1, name, kind, t, d) {
    }

public:
    template <typename... Args, typename T2 = T, typename std::enable_if<!std::is_array<T2>::value>::type * = nullptr>
    FuncRef operator()(Args&&... args) const {
        return get_values<ValueType>().at(0)(std::forward<Args>(args)...);
    }

    template <typename ExprOrVar, typename T2 = T, typename std::enable_if<!std::is_array<T2>::value>::type * = nullptr>
    FuncRef operator()(std::vector<ExprOrVar> args) const {
        return get_values<ValueType>().at(0)(args);
    }

    template <typename T2 = T, typename std::enable_if<!std::is_array<T2>::value>::type * = nullptr>
    operator Func() const {
        return get_values<ValueType>().at(0);
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    size_t size() const {
        return get_values<ValueType>().size();
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    const ValueType &operator[](size_t i) const {
        return get_values<ValueType>()[i];
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    const ValueType &at(size_t i) const {
        return get_values<ValueType>().at(i);
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    typename std::vector<ValueType>::const_iterator begin() const {
        return get_values<ValueType>().begin();
    }

    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    typename std::vector<ValueType>::const_iterator end() const {
        return get_values<ValueType>().end();
    }

    template <typename T2 = T, typename std::enable_if<
        // Only allow T2[]
        std::is_array<T2>::value && std::rank<T2>::value == 1 && std::extent<T2, 0>::value == 0
    >::type * = nullptr>
    void resize(size_t size) {
        GeneratorOutputBase::resize(size);
    }
};

template<typename T>
class GeneratorOutput_Buffer : public GeneratorOutputImpl<T>, public Constrainable {
private:
    using Super = GeneratorOutputImpl<T>;

protected:
    using TBase = typename Super::TBase;

protected:
    GeneratorOutput_Buffer(const std::string &name)
        : Super(name, IOKind::Buffer,
                T::has_static_halide_type ? std::vector<Type>{ T::static_halide_type() } : std::vector<Type>{},
                -1) {
    }

    GeneratorOutput_Buffer(const std::string &name, const std::vector<Type> &t, int d = -1)
        : Super(name, IOKind::Buffer,
                T::has_static_halide_type ? std::vector<Type>{ T::static_halide_type() } : t,
                d) {
        if (T::has_static_halide_type) {
            user_assert(t.empty()) << "Cannot use pass a Type argument for a Buffer with a non-void static type\n";
        } else {
            user_assert(t.size() <= 1) << "Output<Buffer<>>(" << name << ") requires at most one Type, but has " << t.size() << "\n";
        }
    }

    GeneratorOutput_Buffer(const std::string &name, int d)
        : Super(name, IOKind::Buffer, std::vector<Type>{ T::static_halide_type() }, d) {
        static_assert(T::has_static_halide_type, "Must pass a Type argument for a Buffer with a static type of void");
    }

    NO_INLINE std::string get_c_type() const override {
        if (T::has_static_halide_type) {
            return "Halide::Internal::StubOutputBuffer<" +
                halide_type_to_c_type(T::static_halide_type()) +
                ">";
        } else {
            return "Halide::Internal::StubOutputBuffer<>";
        }
    }

    Parameter parameter() const override {
        internal_assert(this->funcs().size() == 1);
        return this->funcs().at(0).output_buffer().parameter();
    }

public:

    // Allow assignment from a Buffer<> to an Output<Buffer<>>;
    // this allows us to use a statically-compiled buffer inside a Generator
    // to assign to an output.
    // TODO: This used to take the buffer as a const ref. This no longer works as
    // using it in a Pipeline might change the dev field so it is currently
    // not considered const. We should consider how this really ought to work.
    template<typename T2>
    NO_INLINE GeneratorOutput_Buffer<T> &operator=(Buffer<T2> &buffer) {
        this->check_value_writable();

        user_assert(T::can_convert_from(buffer))
            << "Cannot assign to the Output \"" << this->name()
            << "\": the expression is not convertible to the same Buffer type and/or dimensions.\n";

        if (this->types_defined()) {
            user_assert(Type(buffer.type()) == this->type())
                << "Output should have type=" << this->type() << " but saw type=" << Type(buffer.type()) << "\n";
        }
        if (this->dims_defined()) {
            user_assert(buffer.dimensions() == this->dims())
                << "Output should have dim=" << this->dims() << " but saw dim=" << buffer.dimensions() << "\n";
        }

        internal_assert(this->exprs_.empty() && this->funcs_.size() == 1);
        user_assert(!this->funcs_.at(0).defined());
        this->funcs_.at(0)(_) = buffer(_);

        return *this;
    }

    // Allow assignment from a StubOutputBuffer to an Output<Buffer>;
    // this allows us to pipeline the results of a Stub to the results
    // of the enclosing Generator.
    template<typename T2>
    NO_INLINE GeneratorOutput_Buffer<T> &operator=(const StubOutputBuffer<T2> &stub_output_buffer) {
        this->check_value_writable();

        const auto &f = stub_output_buffer.f;
        internal_assert(f.defined());

        const auto &output_types = f.output_types();
        user_assert(output_types.size() == 1)
            << "Output should have size=1 but saw size=" << output_types.size() << "\n";

        Buffer<> other(output_types.at(0), nullptr, std::vector<int>(f.dimensions(), 1));
        user_assert(T::can_convert_from(other))
            << "Cannot assign to the Output \"" << this->name()
            << "\": the expression is not convertible to the same Buffer type and/or dimensions.\n";

        if (this->types_defined()) {
            user_assert(output_types.at(0) == this->type())
                << "Output should have type=" << this->type() << " but saw type=" << output_types.at(0) << "\n";
        }
        if (this->dims_defined()) {
            user_assert(f.dimensions() == this->dims())
                << "Output should have dim=" << this->dims() << " but saw dim=" << f.dimensions() << "\n";
        }

        internal_assert(this->exprs_.empty() && this->funcs_.size() == 1);
        user_assert(!this->funcs_.at(0).defined());
        this->funcs_[0] = f;

        return *this;
    }

    GeneratorOutput_Buffer<T> &estimate(Var var, Expr min, Expr extent) {
        internal_assert(this->exprs_.empty() && this->funcs_.size() == 1);
        this->funcs_.at(0).estimate(var, min, extent);
        return *this;
    }
};


template<typename T>
class GeneratorOutput_Func : public GeneratorOutputImpl<T> {
private:
    using Super = GeneratorOutputImpl<T>;

    NO_INLINE Func &get_assignable_func_ref(size_t i) {
        internal_assert(this->exprs_.empty() && this->funcs_.size() > i);
        return this->funcs_.at(i);
    }

protected:
    using TBase = typename Super::TBase;

protected:
    GeneratorOutput_Func(const std::string &name, const std::vector<Type> &t, int d)
        : Super(name, IOKind::Function, t, d) {
    }

    GeneratorOutput_Func(size_t array_size, const std::string &name, const std::vector<Type> &t, int d)
        : Super(array_size, name, IOKind::Function, t, d) {
    }

public:
    // Allow Output<Func> = Func
    template <typename T2 = T, typename std::enable_if<!std::is_array<T2>::value>::type * = nullptr>
    GeneratorOutput_Func<T> &operator=(const Func &f) {
        this->check_value_writable();

        // Don't bother verifying the Func type, dimensions, etc., here:
        // That's done later, when we produce the pipeline.
        get_assignable_func_ref(0) = f;
        return *this;
    }

    // Allow Output<Func[]> = Func
    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    Func &operator[](size_t i) {
        this->check_value_writable();
        return get_assignable_func_ref(i);
    }

    // Allow Func = Output<Func[]>
    template <typename T2 = T, typename std::enable_if<std::is_array<T2>::value>::type * = nullptr>
    const Func &operator[](size_t i) const {
        return Super::operator[](i);
    }

    GeneratorOutput_Func<T> &estimate(Var var, Expr min, Expr extent) {
        internal_assert(this->exprs_.empty() && this->funcs_.size() > 0);
        for (Func &f : this->funcs_) {
            f.estimate(var, min, extent);
        }
        return *this;
    }
};


template<typename T>
class GeneratorOutput_Arithmetic : public GeneratorOutputImpl<T> {
private:
    using Super = GeneratorOutputImpl<T>;
protected:
    using TBase = typename Super::TBase;

protected:
    explicit GeneratorOutput_Arithmetic(const std::string &name)
        : Super(name, IOKind::Function, {type_of<TBase>()}, 0) {
    }

    GeneratorOutput_Arithmetic(size_t array_size, const std::string &name)
        : Super(array_size, name, IOKind::Function, {type_of<TBase>()}, 0) {
    }
};

template<typename T, typename TBase = typename std::remove_all_extents<T>::type>
using GeneratorOutputImplBase =
    typename select_type<
        cond<has_static_halide_type_method<TBase>::value, GeneratorOutput_Buffer<T>>,
        cond<std::is_same<TBase, Func>::value, GeneratorOutput_Func<T>>,
        cond<std::is_arithmetic<TBase>::value, GeneratorOutput_Arithmetic<T>>
    >::type;

}  // namespace Internal

template <typename T>
class GeneratorOutput : public Internal::GeneratorOutputImplBase<T> {
private:
    using Super = Internal::GeneratorOutputImplBase<T>;
protected:
    using TBase = typename Super::TBase;

public:
    explicit GeneratorOutput(const std::string &name)
        : Super(name) {
    }

    explicit GeneratorOutput(const char *name)
        : GeneratorOutput(std::string(name)) {
    }

    GeneratorOutput(size_t array_size, const std::string &name)
        : Super(array_size, name) {
    }

    GeneratorOutput(const std::string &name, int d)
        : Super(name, {}, d) {
    }

    GeneratorOutput(const std::string &name, const Type &t, int d)
        : Super(name, {t}, d) {
    }

    GeneratorOutput(const std::string &name, const std::vector<Type> &t, int d)
        : Super(name, t, d) {
    }

    GeneratorOutput(size_t array_size, const std::string &name, int d)
        : Super(array_size, name, {}, d) {
    }

    GeneratorOutput(size_t array_size, const std::string &name, const Type &t, int d)
        : Super(array_size, name, {t}, d) {
    }

    GeneratorOutput(size_t array_size, const std::string &name, const std::vector<Type> &t, int d)
        : Super(array_size, name, t, d) {
    }

    // TODO: This used to take the buffer as a const ref. This no longer works as
    // using it in a Pipeline might change the dev field so it is currently
    // not considered const. We should consider how this really ought to work.
    template <typename T2>
    GeneratorOutput<T> &operator=(Buffer<T2> &buffer) {
        Super::operator=(buffer);
        return *this;
    }

    template <typename T2>
    GeneratorOutput<T> &operator=(const Internal::StubOutputBuffer<T2> &stub_output_buffer) {
        Super::operator=(stub_output_buffer);
        return *this;
    }

    GeneratorOutput<T> &operator=(const Func &f) {
        Super::operator=(f);
        return *this;
    }
};

namespace Internal {

template<typename T>
T parse_scalar(const std::string &value) {
    std::istringstream iss(value);
    T t;
    iss >> t;
    user_assert(!iss.fail() && iss.get() == EOF) << "Unable to parse: " << value;
    return t;
}

EXPORT std::vector<Type> parse_halide_type_list(const std::string &types);

// This is a type of GeneratorParam used internally to create 'synthetic' params
// (e.g. image.type, image.dim); it is not possible for user code to instantiate it.
template<typename T>
class GeneratorParam_Synthetic : public GeneratorParamImpl<T> {
public:
    void set_from_string(const std::string &new_value_string) override {
        set_from_string_impl<T>(new_value_string);
    }

    std::string to_string() const override {
        internal_error;
        return std::string();
    }

    std::string call_to_string(const std::string &v) const override {
        internal_error;
        return std::string();
    }

    std::string get_c_type() const override {
        internal_error;
        return std::string();
    }

    bool is_synthetic_param() const override {
        return true;
    }

private:
    friend class GeneratorBase;

    enum Which { Type, Dim, ArraySize };
    GeneratorParam_Synthetic(const std::string &name, GIOBase &gio, Which which) : GeneratorParamImpl<T>(name, T()), gio(gio), which(which) {}

    template <typename T2 = T, typename std::enable_if<std::is_same<T2, ::Halide::Type>::value>::type * = nullptr>
    void set_from_string_impl(const std::string &new_value_string) {
        internal_assert(which == Type);
        gio.types_ = parse_halide_type_list(new_value_string);
    }

    template <typename T2 = T, typename std::enable_if<std::is_integral<T2>::value>::type * = nullptr>
    void set_from_string_impl(const std::string &new_value_string) {
        if (which == Dim) {
            gio.dims_ = parse_scalar<T2>(new_value_string);
        } else if (which == ArraySize) {
            gio.array_size_ = parse_scalar<T2>(new_value_string);
        } else {
            internal_error;
        }
    }

    GIOBase &gio;
    const Which which;
};


class GeneratorStub;

}  // namespace Internal

/** GeneratorContext is a base class that is used when using Generators (or Stubs) directly;
 * it is used to allow the outer context (typically, either a Generator or "top-level" code)
 * to specify certain information to the inner context to ensure that inner and outer
 * Generators are compiled in a compatible way.
 *
 * If you are using this at "top level" (e.g. with the JIT), you can construct a GeneratorContext
 * with a Target:
 * \code
 *   auto my_stub = MyStub(
 *       GeneratorContext(get_target_from_environment()),
 *       // inputs
 *       { ... },
 *       // generator params
 *       { ... }
 *   );
 * \endcode
 *
 * Note that all Generators inherit from GeneratorContext, so if you are using a Stub
 * from within a Generator, you can just pass 'this' for the GeneratorContext:
 * \code
 *  struct SomeGen : Generator<SomeGen> {
 *   void generate() {
 *     ...
 *     auto my_stub = MyStub(
 *       this,  // GeneratorContext
 *       // inputs
 *       { ... },
 *       // generator params
 *       { ... }
 *     );
 *     ...
 *   }
 *  };
 * \endcode
 */
class GeneratorContext {
public:
    using ExternsMap = std::map<std::string, ExternalCode>;

    explicit GeneratorContext(const Target &t) :
        target("target", t),
        externs_map(std::make_shared<ExternsMap>()),
        value_tracker(std::make_shared<Internal::ValueTracker>()) {}
    virtual ~GeneratorContext() {}

    inline Target get_target() const { return target; }

    /** Generators can register ExternalCode objects onto
     * themselves. The Generator infrastructure will arrange to have
     * this ExternalCode appended to the Module that is finally
     * compiled using the Generator. This allows encapsulating
     * functionality that depends on external libraries or handwritten
     * code for various targets. The name argument should match the
     * name of the ExternalCode block and is used to ensure the same
     * code block is not duplicated in the output. Halide does not do
     * anything other than to compare names for equality. To guarantee
     * uniqueness in public code, we suggest using a Java style
     * inverted domain name followed by organization specific
     * naming. E.g.:
     *     com.yoyodyne.overthruster.0719acd19b66df2a9d8d628a8fefba911a0ab2b7
     *
     * See test/generator/external_code_generator.cpp for example use. */
    inline std::shared_ptr<ExternsMap> get_externs_map() const {
        return externs_map;
    }

    template <typename T>
    inline std::unique_ptr<T> create() const {
        return T::create(*this);
    }

    template <typename T, typename... Args>
    inline std::unique_ptr<T> apply(const Args &...args) const {
        auto t = this->create<T>();
        t->apply(args...);
        return t;
    }

protected:
    GeneratorParam<Target> target;
    std::shared_ptr<ExternsMap> externs_map;
    std::shared_ptr<Internal::ValueTracker> value_tracker;

    GeneratorContext() : GeneratorContext(Target()) {}

    inline void init_from_context(const Halide::GeneratorContext &context) {
        target.set(context.get_target());
        value_tracker = context.get_value_tracker();
        externs_map = context.get_externs_map();
    }

    inline std::shared_ptr<Internal::ValueTracker> get_value_tracker() const {
        return value_tracker;
    }

    // No copy
    GeneratorContext(const GeneratorContext &) = delete;
    void operator=(const GeneratorContext &) = delete;
    // No move
    GeneratorContext(GeneratorContext&&) = delete;
    void operator=(GeneratorContext&&) = delete;
};

class NamesInterface {
    // Names in this class are only intended for use in derived classes.
protected:
    // Import a consistent list of Halide names that can be used in
    // Halide generators without qualification.
    using Expr = Halide::Expr;
    using ExternFuncArgument = Halide::ExternFuncArgument;
    using Func = Halide::Func;
    using GeneratorContext = Halide::GeneratorContext;
    using ImageParam = Halide::ImageParam;
    using LoopLevel = Halide::LoopLevel;
    using Pipeline = Halide::Pipeline;
    using RDom = Halide::RDom;
    using TailStrategy = Halide::TailStrategy;
    using Target = Halide::Target;
    using Tuple = Halide::Tuple;
    using Type = Halide::Type;
    using Var = Halide::Var;
    using NameMangling = Halide::NameMangling;
    template <typename T> static Expr cast(Expr e) { return Halide::cast<T>(e); }
    static inline Expr cast(Halide::Type t, Expr e) { return Halide::cast(t, e); }
    template <typename T> using GeneratorParam = Halide::GeneratorParam<T>;
    template <typename T> using ScheduleParam = Halide::ScheduleParam<T>;
    template <typename T = void> using Buffer = Halide::Buffer<T>;
    template <typename T> using Param = Halide::Param<T>;
    static inline Type Bool(int lanes = 1) { return Halide::Bool(lanes); }
    static inline Type Float(int bits, int lanes = 1) { return Halide::Float(bits, lanes); }
    static inline Type Int(int bits, int lanes = 1) { return Halide::Int(bits, lanes); }
    static inline Type UInt(int bits, int lanes = 1) { return Halide::UInt(bits, lanes); }
};

namespace Internal {

template<typename ...Args>
struct NoRealizations : std::false_type {};

template<>
struct NoRealizations<> : std::true_type {};

template<typename T, typename ...Args>
struct NoRealizations<T, Args...> {
    static const bool value = !std::is_convertible<T, Realization>::value && NoRealizations<Args...>::value;
};

class GeneratorStub;
class SimpleGeneratorFactory;

// Note that these functions must never return null:
// if they cannot return a valid Generator, they must assert-fail.
using GeneratorFactory = std::function<std::unique_ptr<GeneratorBase>(const GeneratorContext&)>;

class GeneratorBase : public NamesInterface, public GeneratorContext {
public:
    struct EmitOptions {
        bool emit_o, emit_h, emit_cpp, emit_assembly, emit_bitcode, emit_stmt, emit_stmt_html, emit_static_library, emit_cpp_stub;
        // This is an optional map used to replace the default extensions generated for
        // a file: if an key matches an output extension, emit those files with the
        // corresponding value instead (e.g., ".s" -> ".assembly_text"). This is
        // empty by default; it's mainly useful in build environments where the default
        // extensions are problematic, and avoids the need to rename output files
        // after the fact.
        std::map<std::string, std::string> substitutions;
        EmitOptions()
            : emit_o(false), emit_h(true), emit_cpp(false), emit_assembly(false),
              emit_bitcode(false), emit_stmt(false), emit_stmt_html(false), emit_static_library(true), emit_cpp_stub(false) {}
    };

    EXPORT virtual ~GeneratorBase();

    EXPORT void set_generator_param(const std::string &name, const std::string &value);
    EXPORT void set_generator_and_schedule_param_values(const std::map<std::string, std::string> &params);

    template<typename T>
    GeneratorBase &set_generator_param(const std::string &name, const T &value) {
        find_generator_param_by_name(name).set(value);
        return *this;
    }

    template<typename T>
    GeneratorBase &set_schedule_param(const std::string &name, const T &value) {
        find_schedule_param_by_name(name).set(value);
        return *this;
    }

    /** Given a data type, return an estimate of the "natural" vector size
     * for that data type when compiling for the current target. */
    int natural_vector_size(Halide::Type t) const {
        return get_target().natural_vector_size(t);
    }

    /** Given a data type, return an estimate of the "natural" vector size
     * for that data type when compiling for the current target. */
    template <typename data_t>
    int natural_vector_size() const {
        return get_target().natural_vector_size<data_t>();
    }

    EXPORT void emit_cpp_stub(const std::string &stub_file_path);

    // Call build() and produce a Module for the result.
    // If function_name is empty, generator_name() will be used for the function.
    EXPORT Module build_module(const std::string &function_name = "",
                               const LoweredFunc::LinkageType linkage_type = LoweredFunc::ExternalPlusMetadata);

    /**
     * set_inputs is a variadic wrapper around set_inputs_vector, which makes usage much simpler
     * in many cases, as it constructs the relevant entries for the vector for you, which
     * is often a bit unintuitive at present. The arguments are passed in Input<>-declaration-order,
     * and the types must be compatible. Array inputs are passed as std::vector<> of the relevant type.
     *
     * Note: at present, scalar input types must match *exactly*, i.e., for Input<uint8_t>, you
     * must pass an argument that is actually uint8_t; an argument that is int-that-will-fit-in-uint8
     * will assert-fail at Halide compile time.
     */
    template <typename... Args>
    void set_inputs(const Args &...args) {
        // set_inputs_vector() checks this too, but checking it here allows build_inputs() to avoid out-of-range checks.
        ParamInfo &pi = param_info();
        user_assert(sizeof...(args) == pi.filter_inputs.size())
                << "Expected exactly " << pi.filter_inputs.size()
                << " inputs but got " << sizeof...(args) << "\n";
        set_inputs_vector(build_inputs(std::forward_as_tuple<const Args &...>(args...), make_index_sequence<sizeof...(Args)>{}));
    }

    Realization realize(std::vector<int32_t> sizes) {
        check_scheduled("realize");
        return get_pipeline().realize(sizes, get_target());
    }

    // Only enable if none of the args are Realization; otherwise we can incorrectly
    // select this method instead of the Realization-as-outparam variant
    template <typename... Args, typename std::enable_if<NoRealizations<Args...>::value>::type * = nullptr>
    Realization realize(Args&&... args) {
        check_scheduled("realize");
        return get_pipeline().realize(std::forward<Args>(args)..., get_target());
    }

    void realize(Realization r) {
        check_scheduled("realize");
        get_pipeline().realize(r, get_target());
    }

    // Return the Pipeline that has been built by the generate() method.
    // This method can only be used from a Generator that has a generate()
    // method (vs a build() method), and currently can only be called from
    // the schedule() method. (This may be relaxed in the future to allow
    // calling from generate() as long as all Outputs have been defined.)
    EXPORT Pipeline get_pipeline();

    /** Generate a schedule for the Generator's pipeline. */
    //@{
    EXPORT std::string auto_schedule_outputs(const MachineParams &arch_params);
    EXPORT std::string auto_schedule_outputs();
    //@}

protected:
    EXPORT GeneratorBase(size_t size, const void *introspection_helper);
    EXPORT void set_generator_names(const std::string &registered_name, const std::string &stub_name);

    EXPORT virtual Pipeline build_pipeline() = 0;
    EXPORT virtual void call_generate() = 0;
    EXPORT virtual void call_schedule() = 0;

    EXPORT void track_parameter_values(bool include_outputs);

    EXPORT void pre_build();
    EXPORT void post_build();
    EXPORT void pre_generate();
    EXPORT void post_generate();
    EXPORT void pre_schedule();
    EXPORT void post_schedule();

    template<typename T>
    using Input = GeneratorInput<T>;

    template<typename T>
    using Output = GeneratorOutput<T>;

    template<typename T>
    using ScheduleParam = ScheduleParam<T>;

    // A Generator's creation and usage must go in a certain phase to ensure correctness;
    // the state machine here is advanced and checked at various points to ensure
    // this is the case.
    enum Phase {
        // Generator has just come into being.
        Created,

        // All Input<>/Param<> fields have been set. (Applicable only in JIT mode;
        // in AOT mode, this can be skipped, going Created->GenerateCalled directly.)
        InputsSet,

        // Generator has had its generate() method called. (For Generators with
        // a build() method instead of generate(), this phase will be skipped
        // and will advance directly to ScheduleCalled.)
        GenerateCalled,

        // Generator has had its schedule() method (if any) called.
        ScheduleCalled,
    } phase{Created};

    void check_exact_phase(Phase expected_phase) const;
    void check_min_phase(Phase expected_phase) const;
    void advance_phase(Phase new_phase);

private:
    friend void ::Halide::Internal::generator_test();
    friend class GeneratorParamBase;
    friend class GeneratorInputBase;
    friend class GeneratorOutputBase;
    friend class GeneratorStub;
    friend class SimpleGeneratorFactory;
    friend class StubOutputBufferBase;

    struct ParamInfo {
        EXPORT ParamInfo(GeneratorBase *generator, const size_t size);

        // Ordered-list of non-null ptrs to GeneratorParam<> fields.
        std::vector<Internal::GeneratorParamBase *> generator_params;

        // Ordered-list of non-null ptrs to ScheduleParam<> fields.
        std::vector<Internal::ScheduleParamBase *> schedule_params;

        // Ordered-list of non-null ptrs to Input<> fields.
        // Only one of filter_inputs and filter_params may be nonempty.
        std::vector<Internal::GeneratorInputBase *> filter_inputs;

        // Ordered-list of non-null ptrs to Param<> or ImageParam<> fields.
        // Must be empty if the Generator has a build() method rather than generate()/schedule().
        // Only one of filter_inputs and filter_params may be nonempty.
        std::vector<Internal::Parameter *> filter_params;

        // Ordered-list of non-null ptrs to Output<> fields; empty if old-style Generator.
        std::vector<Internal::GeneratorOutputBase *> filter_outputs;

        // Convenience structure to look up GP by name.
        std::map<std::string, Internal::GeneratorParamBase *> generator_params_by_name;

        // Convenience structure to look up SP by name.
        std::map<std::string, Internal::ScheduleParamBase *> schedule_params_by_name;

    private:
        // list of synthetic GP's that we dynamically created; this list only exists to simplify
        // lifetime management, and shouldn't be accessed directly outside of our ctor/dtor,
        // regardless of friend access.
        std::vector<std::unique_ptr<Internal::GeneratorParamBase>> owned_synthetic_params;
    };

    const size_t size;
    // Lazily-allocated-and-inited struct with info about our various Params.
    // Do not access directly: use the param_info() getter to lazy-init.
    std::unique_ptr<ParamInfo> param_info_ptr;

    mutable std::shared_ptr<ExternsMap> externs_map;

    bool inputs_set{false};
    std::string generator_registered_name, generator_stub_name;
    Pipeline pipeline;

    // Return our ParamInfo (lazy-initing as needed).
    EXPORT ParamInfo &param_info();

    EXPORT Internal::GeneratorParamBase &find_generator_param_by_name(const std::string &name);
    EXPORT Internal::ScheduleParamBase &find_schedule_param_by_name(const std::string &name);

    EXPORT void check_scheduled(const char* m) const;

    EXPORT void build_params(bool force = false);

    // Provide private, unimplemented, wrong-result-type methods here
    // so that Generators don't attempt to call the global methods
    // of the same name by accident: use the get_target() method instead.
    void get_host_target();
    void get_jit_target_from_environment();
    void get_target_from_environment();

    EXPORT Func get_first_output();
    EXPORT Func get_output(const std::string &n);
    EXPORT std::vector<Func> get_output_vector(const std::string &n);

    EXPORT void set_inputs_vector(const std::vector<std::vector<StubInput>> &inputs);

    EXPORT static void check_input_is_singular(Internal::GeneratorInputBase *in);
    EXPORT static void check_input_is_array(Internal::GeneratorInputBase *in);
    EXPORT static void check_input_kind(Internal::GeneratorInputBase *in, Internal::IOKind kind);

    // Allow Buffer<> if:
    // -- we are assigning it to an Input<Buffer<>> (with compatible type and dimensions),
    // causing the Input<Buffer<>> to become a precompiled buffer in the generated code.
    // -- we are assigningit to an Input<Func>, in which case we just Func-wrap the Buffer<>.
    template<typename T>
    std::vector<StubInput> build_input(size_t i, const Buffer<T> &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_is_singular(in);
        const auto k = in->kind();
        if (k == Internal::IOKind::Buffer) {
            Halide::Buffer<> b = arg;
            StubInputBuffer<> sib(b);
            StubInput si(sib);
            return {si};
        } else if (k == Internal::IOKind::Function) {
            Halide::Func f(arg.name() + "_im");
            f(Halide::_) = arg(Halide::_);
            StubInput si(f);
            return {si};
        } else {
            check_input_kind(in, Internal::IOKind::Buffer);  // just to trigger assertion
            return {};
        }
    }

    // Allow Input<Buffer<>> if:
    // -- we are assigning it to another Input<Buffer<>> (with compatible type and dimensions),
    // allowing us to simply pipe a parameter from an enclosing Generator to the Invoker.
    // -- we are assigningit to an Input<Func>, in which case we just Func-wrap the Input<Buffer<>>.
    template<typename T>
    std::vector<StubInput> build_input(size_t i, const GeneratorInput<Buffer<T>> &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_is_singular(in);
        const auto k = in->kind();
        if (k == Internal::IOKind::Buffer) {
            StubInputBuffer<> sib = arg;
            StubInput si(sib);
            return {si};
        } else if (k == Internal::IOKind::Function) {
            Halide::Func f = arg.funcs().at(0);
            StubInput si(f);
            return {si};
        } else {
            check_input_kind(in, Internal::IOKind::Buffer);  // just to trigger assertion
            return {};
        }
    }

    // Allow Func iff we are assigning it to an Input<Func> (with compatible type and dimensions).
    std::vector<StubInput> build_input(size_t i, const Func &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_kind(in, Internal::IOKind::Function);
        check_input_is_singular(in);
        Halide::Func f = arg;
        StubInput si(f);
        return {si};
    }

    // Allow vector<Func> iff we are assigning it to an Input<Func[]> (with compatible type and dimensions).
    std::vector<StubInput> build_input(size_t i, const std::vector<Func> &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_kind(in, Internal::IOKind::Function);
        check_input_is_array(in);
        // My kingdom for a list comprehension...
        std::vector<StubInput> siv;
        siv.reserve(arg.size());
        for (const auto &f : arg) {
            siv.emplace_back(f);
        }
        return siv;
    }

    // Expr must be Input<Scalar>.
    std::vector<StubInput> build_input(size_t i, const Expr &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_kind(in, Internal::IOKind::Scalar);
        check_input_is_singular(in);
        StubInput si(arg);
        return {si};
    }

    // (Array form)
    std::vector<StubInput> build_input(size_t i, const std::vector<Expr> &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_kind(in, Internal::IOKind::Scalar);
        check_input_is_array(in);
        std::vector<StubInput> siv;
        siv.reserve(arg.size());
        for (const auto &value : arg) {
            siv.emplace_back(value);
        }
        return siv;
    }

    // Any other type must be convertible to Expr and must be associated with an Input<Scalar>.
    // Use is_arithmetic since some Expr conversions are explicit.
    template<typename T,
             typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    std::vector<StubInput> build_input(size_t i, const T &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_kind(in, Internal::IOKind::Scalar);
        check_input_is_singular(in);
        // We must use an explicit Expr() ctor to preserve the type
        Expr e(arg);
        StubInput si(e);
        return {si};
    }

    // (Array form)
    template<typename T,
             typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    std::vector<StubInput> build_input(size_t i, const std::vector<T> &arg) {
        auto *in = param_info().filter_inputs.at(i);
        check_input_kind(in, Internal::IOKind::Scalar);
        check_input_is_array(in);
        std::vector<StubInput> siv;
        siv.reserve(arg.size());
        for (const auto &value : arg) {
            // We must use an explicit Expr() ctor to preserve the type;
            // otherwise, implicit conversions can downgrade (e.g.) float -> int
            Expr e(value);
            siv.emplace_back(e);
        }
        return siv;
    }

    template<typename... Args, size_t... Indices>
    std::vector<std::vector<StubInput>> build_inputs(const std::tuple<const Args &...>& t, index_sequence<Indices...>) {
        return {build_input(Indices, std::get<Indices>(t))...};
    }

    // No copy
    GeneratorBase(const GeneratorBase &) = delete;
    void operator=(const GeneratorBase &) = delete;
    // No move
    GeneratorBase(GeneratorBase&& that) = delete;
    void operator=(GeneratorBase&& that) = delete;
};

class GeneratorRegistry {
public:
    EXPORT static void register_factory(const std::string &name, GeneratorFactory generator_factory);
    EXPORT static void unregister_factory(const std::string &name);
    EXPORT static std::vector<std::string> enumerate();
    // Note that this method will never return null:
    // if it cannot return a valid Generator, it should assert-fail.
    EXPORT static std::unique_ptr<GeneratorBase> create(const std::string &name,
                                                        const GeneratorContext &context);

private:
    using GeneratorFactoryMap = std::map<const std::string, GeneratorFactory>;

    GeneratorFactoryMap factories;
    std::mutex mutex;

    EXPORT static GeneratorRegistry &get_registry();

    GeneratorRegistry() {}
    GeneratorRegistry(const GeneratorRegistry &) = delete;
    void operator=(const GeneratorRegistry &) = delete;
};

}  // namespace Internal

template <class T>
class Generator : public Internal::GeneratorBase {
protected:

    // TODO: This causes problems for existing code that declares helper
    // methods (that use ImageParam, etc as arguments) outside the class body,
    // as there is an ambiguity between Halide::ImageParam and Generator<T>::ImageParam.
    //
    // Consider re-enabling this at some point in the future when the likelihood of
    // collision with existing code is much smaller.
    //
    // // Add wrapper types here that exists just to allow us to tag
    // // ImageParam/Param-used-inside-Generator with HALIDE_ATTRIBUTE_DEPRECATED.
    // // (This won't catch code that uses "Halide::Param" or "Halide::ImageParam"
    // // but those are somewhat uncommon cases.)

    // template<typename T2>
    // class Param : public ::Halide::Param<T2> {
    // public:
    //     template <typename... Args>
    //     HALIDE_ATTRIBUTE_DEPRECATED("Using Param<> in Generators is deprecated; please use Input<> instead.")
    //     explicit Param(const Args &...args) : ::Halide::Param<T2>(args...) { }
    // };

    // class ImageParam : public ::Halide::ImageParam {
    // public:
    //     template <typename... Args>
    //     HALIDE_ATTRIBUTE_DEPRECATED("Using ImageParam<> in Generators is deprecated; please use Input<Buffer<>> instead.")
    //     explicit ImageParam(const Args &...args) : ::Halide::ImageParam(args...) { }
    // };

protected:
    Generator() :
        Internal::GeneratorBase(sizeof(T),
                                Internal::Introspection::get_introspection_helper<T>()) {}

public:
    static std::unique_ptr<T> create(const Halide::GeneratorContext &context) {
        // We must have an object of type T (not merely GeneratorBase) to call a protected method,
        // because CRTP is a weird beast.
        auto g = std::unique_ptr<T>(new T());
        g->init_from_context(context);
        return g;
    }

    // This is public but intended only for use by the HALIDE_REGISTER_GENERATOR() macro.
    static std::unique_ptr<T> create(const Halide::GeneratorContext &context,
                                     const std::string &registered_name,
                                     const std::string &stub_name) {
        auto g = create(context);
        g->set_generator_names(registered_name, stub_name);
        return g;
    }

    using Internal::GeneratorBase::apply;
    using Internal::GeneratorBase::create;

    template <typename... Args>
    void apply(const Args &...args) {
        static_assert(has_generate_method<T>::value && has_schedule_method<T>::value,
            "apply() is not supported for old-style Generators.");
        set_inputs(args...);
        call_generate();
        call_schedule();
    }

private:

    // std::is_member_function_pointer will fail if there is no member of that name,
    // so we use a little SFINAE to detect if there are method-shaped members.
    template<typename>
    struct type_sink { typedef void type; };

    template<typename T2, typename = void>
    struct has_generate_method : std::false_type {};

    template<typename T2>
    struct has_generate_method<T2, typename type_sink<decltype(std::declval<T2>().generate())>::type> : std::true_type {};

    template<typename T2, typename = void>
    struct has_schedule_method : std::false_type {};

    template<typename T2>
    struct has_schedule_method<T2, typename type_sink<decltype(std::declval<T2>().schedule())>::type> : std::true_type {};

    template <typename T2 = T,
              typename std::enable_if<!has_generate_method<T2>::value>::type * = nullptr>

    // Implementations for build_pipeline_impl(), specialized on whether we
    // have build() or generate()/schedule() methods.

    // MSVC apparently has some weirdness with the usual sfinae tricks
    // for detecting method-shaped things, so we can't actually use
    // the helpers above outside of static_assert. Instead we make as
    // many overloads as we can exist, and then use C++'s preference
    // for treating a 0 as an int rather than a double to choose one
    // of them.
    Pipeline build_pipeline_impl(double) {
        static_assert(!has_schedule_method<T2>::value, "The schedule() method is ignored if you define a build() method; use generate() instead.");
        pre_build();
        Pipeline p = ((T *)this)->build();
        post_build();
        return p;
    }

    template <typename T2 = T,
              typename = decltype(std::declval<T2>().generate())>
    Pipeline build_pipeline_impl(int) {
        ((T *)this)->call_generate_impl(0);
        ((T *)this)->call_schedule_impl(0, 0);
        return get_pipeline();
    }

    // Implementations for call_generate_impl(), specialized on whether we
    // have build() or generate()/schedule() methods.

    void call_generate_impl(double) {
        user_error << "Unimplemented";
    }

    template <typename T2 = T,
              typename = decltype(std::declval<T2>().generate())>
    void call_generate_impl(int) {
        T *t = (T*)this;
        static_assert(std::is_void<decltype(t->generate())>::value, "generate() must return void");
        pre_generate();
        t->generate();
        post_generate();
    }

    // Implementations for call_schedule_impl(), specialized on whether we
    // have build() or generate()/schedule() methods.

    void call_schedule_impl(double, double) {
        user_error << "Unimplemented";
    }

    template<typename T2 = T,
             typename = decltype(std::declval<T2>().generate())>
    void call_schedule_impl(double, int) {
        // Generator has a generate() method but no schedule() method. This is ok. Just advance the phase.
        pre_schedule();
        post_schedule();
    }

    template<typename T2 = T,
             typename = decltype(std::declval<T2>().generate()),
             typename = decltype(std::declval<T2>().schedule())>
    void call_schedule_impl(int, int) {
        T *t = (T*)this;
        static_assert(std::is_void<decltype(t->schedule())>::value, "schedule() must return void");
        pre_schedule();
        t->schedule();
        post_schedule();
    }

protected:
    Pipeline build_pipeline() override {
        return this->build_pipeline_impl(0);
    }

    void call_generate() override {
        this->call_generate_impl(0);
    }

    void call_schedule() override {
        this->call_schedule_impl(0, 0);
    }

private:
    friend void ::Halide::Internal::generator_test();
    friend class Internal::SimpleGeneratorFactory;
    friend void ::Halide::Internal::generator_test();
    friend class ::Halide::GeneratorContext;

    // No copy
    Generator(const Generator &) = delete;
    void operator=(const Generator &) = delete;
    // No move
    Generator(Generator&& that) = delete;
    void operator=(Generator&& that) = delete;
};

namespace Internal {

class RegisterGenerator {
public:
    RegisterGenerator(const char* registered_name, GeneratorFactory generator_factory) {
        Internal::GeneratorRegistry::register_factory(registered_name, generator_factory);
    }
};

class GeneratorStub : public NamesInterface {
public:
    // default ctor
    GeneratorStub() = default;

    // move constructor
    GeneratorStub(GeneratorStub&& that) : generator(std::move(that.generator)) {}

    // move assignment operator
    GeneratorStub& operator=(GeneratorStub&& that) {
        generator = std::move(that.generator);
        return *this;
    }

    Target get_target() const { return generator->get_target(); }

   template<typename T>
   GeneratorStub &set_schedule_param(const std::string &name, const T &value) {
       generator->set_schedule_param(name, value);
       return *this;
   }

   GeneratorStub &schedule() {
       generator->call_schedule();
       return *this;
   }

    // Overloads for first output
    operator Func() const {
        return get_first_output();
    }

    template <typename... Args>
    FuncRef operator()(Args&&... args) const {
        return get_first_output()(std::forward<Args>(args)...);
    }

    template <typename ExprOrVar>
    FuncRef operator()(std::vector<ExprOrVar> args) const {
        return get_first_output()(args);
    }

    Realization realize(std::vector<int32_t> sizes) {
        return generator->realize(sizes);
    }

    // Only enable if none of the args are Realization; otherwise we can incorrectly
    // select this method instead of the Realization-as-outparam variant
    template <typename... Args, typename std::enable_if<NoRealizations<Args...>::value>::type * = nullptr>
    Realization realize(Args&&... args) {
        return generator->realize(std::forward<Args>(args)...);
    }

    void realize(Realization r) {
        generator->realize(r);
    }

    virtual ~GeneratorStub() {}

protected:
    EXPORT GeneratorStub(const GeneratorContext &context,
                  GeneratorFactory generator_factory,
                  const std::map<std::string, std::string> &generator_params,
                  const std::vector<std::vector<Internal::StubInput>> &inputs);

    ScheduleParamBase &get_schedule_param(const std::string &n) const {
        return generator->find_schedule_param_by_name(n);
    }

    // Output(s)
    // TODO: identify vars used
    Func get_output(const std::string &n) const {
        return generator->get_output(n);
    }

    template<typename T2>
    T2 get_output_buffer(const std::string &n) const {
        return T2(get_output(n), generator);
    }

    std::vector<Func> get_output_vector(const std::string &n) const {
        return generator->get_output_vector(n);
    }

    bool has_generator() const {
        return generator != nullptr;
    }

    template<typename Ratio>
    static double ratio_to_double() {
        return (double)Ratio::num / (double)Ratio::den;
    }

    static std::vector<StubInput> to_stub_input_vector(const Expr &e) {
        return { StubInput(e) };
    }

    static std::vector<StubInput> to_stub_input_vector(const Func &f) {
        return { StubInput(f) };
    }

    template<typename T = void>
    static std::vector<StubInput> to_stub_input_vector(const StubInputBuffer<T> &b) {
        return { StubInput(b) };
    }

    template <typename T>
    static std::vector<StubInput> to_stub_input_vector(const std::vector<T> &v) {
        std::vector<StubInput> r;
        std::copy(v.begin(), v.end(), std::back_inserter(r));
        return r;
    }

    EXPORT void verify_same_funcs(const Func &a, const Func &b);
    EXPORT void verify_same_funcs(const std::vector<Func>& a, const std::vector<Func>& b);

    template<typename T2>
    void verify_same_funcs(const StubOutputBuffer<T2> &a, const StubOutputBuffer<T2> &b) {
        verify_same_funcs(a.f, b.f);
    }

private:
    std::shared_ptr<GeneratorBase> generator;

    Func get_first_output() const {
        return generator->get_first_output();
    }
    explicit GeneratorStub(const GeneratorStub &) = delete;
    GeneratorStub &operator=(const GeneratorStub &) = delete;
    explicit GeneratorStub(const GeneratorStub &&) = delete;
    GeneratorStub &operator=(const GeneratorStub &&) = delete;
};

}  // namespace Internal


}  // namespace Halide

// Define this namespace at global scope so that anonymous namespaces won't
// defeat our static_assert check; define a dummy type inside so we can
// check for type aliasing injected by anonymous namespace usage
namespace halide_register_generator {
    struct halide_global_ns;
};

#define _HALIDE_REGISTER_GENERATOR_IMPL(GEN_CLASS_NAME, GEN_REGISTRY_NAME, FULLY_QUALIFIED_STUB_NAME) \
    namespace halide_register_generator { \
        struct halide_global_ns; \
        namespace GEN_REGISTRY_NAME##_ns { \
            std::unique_ptr<Halide::Internal::GeneratorBase> factory(const Halide::GeneratorContext& context) { \
                return GEN_CLASS_NAME::create(context, #GEN_REGISTRY_NAME, #FULLY_QUALIFIED_STUB_NAME); \
            } \
        } \
        static auto reg_##GEN_REGISTRY_NAME = Halide::Internal::RegisterGenerator(#GEN_REGISTRY_NAME, GEN_REGISTRY_NAME##_ns::factory); \
    } \
    static_assert(std::is_same<::halide_register_generator::halide_global_ns, halide_register_generator::halide_global_ns>::value, \
                 "HALIDE_REGISTER_GENERATOR must be used at global scope");

#define _HALIDE_REGISTER_GENERATOR2(GEN_CLASS_NAME, GEN_REGISTRY_NAME) \
    _HALIDE_REGISTER_GENERATOR_IMPL(GEN_CLASS_NAME, GEN_REGISTRY_NAME, GEN_REGISTRY_NAME)

#define _HALIDE_REGISTER_GENERATOR3(GEN_CLASS_NAME, GEN_REGISTRY_NAME, FULLY_QUALIFIED_STUB_NAME) \
    _HALIDE_REGISTER_GENERATOR_IMPL(GEN_CLASS_NAME, GEN_REGISTRY_NAME, FULLY_QUALIFIED_STUB_NAME)

// MSVC has a broken implementation of variadic macros: it expands __VA_ARGS__
// as a single token in argument lists (rather than multiple tokens).
// Jump through some hoops to work around this.
#define __HALIDE_REGISTER_ARGCOUNT_IMPL(_1, _2, _3, COUNT, ...) \
   COUNT

#define _HALIDE_REGISTER_ARGCOUNT_IMPL(ARGS) \
   __HALIDE_REGISTER_ARGCOUNT_IMPL ARGS

#define _HALIDE_REGISTER_ARGCOUNT(...) \
   _HALIDE_REGISTER_ARGCOUNT_IMPL((__VA_ARGS__, 3, 2, 1, 0))

#define ___HALIDE_REGISTER_CHOOSER(COUNT) \
    _HALIDE_REGISTER_GENERATOR##COUNT

#define __HALIDE_REGISTER_CHOOSER(COUNT) \
    ___HALIDE_REGISTER_CHOOSER(COUNT)

#define _HALIDE_REGISTER_CHOOSER(COUNT) \
    __HALIDE_REGISTER_CHOOSER(COUNT)

#define _HALIDE_REGISTER_GENERATOR_PASTE(A, B) \
    A B

#define HALIDE_REGISTER_GENERATOR(...) \
    _HALIDE_REGISTER_GENERATOR_PASTE(_HALIDE_REGISTER_CHOOSER(_HALIDE_REGISTER_ARGCOUNT(__VA_ARGS__)), (__VA_ARGS__))

#endif  // HALIDE_GENERATOR_H_
#ifndef HALIDE_HEXAGON_OFFLOAD_H
#define HALIDE_HEXAGON_OFFLOAD_H

/** \file
 * Defines a lowering pass to pull loops marked with the
 * Hexagon device API to a separate module, and call them through the
 * Hexagon host runtime module.
 */


namespace Halide {
namespace Internal {

/** Pull loops marked with the Hexagon device API to a separate
 * module, and call them through the Hexagon host runtime module. */
Stmt inject_hexagon_rpc(Stmt s, const Target &host_target, Module &module);

Buffer<uint8_t> compile_module_to_hexagon_shared_object(const Module &device_code);

}
}

#endif
#ifndef HALIDE_IR_HEXAGON_OPTIMIZE_H
#define HALIDE_IR_HEXAGON_OPTIMIZE_H

/** \file
 * Tools for optimizing IR for Hexagon.
 */


namespace Halide {
namespace Internal {

/** Replace indirect and other loads with simple loads + vlut
 * calls. */
EXPORT Stmt optimize_hexagon_shuffles(Stmt s, int lut_alignment);

/** Generate vtmpy instruction if possible */
EXPORT Stmt vtmpy_generator(Stmt s);

/** Hexagon deinterleaves when performing widening operations, and
 * interleaves when performing narrowing operations. This pass
 * rewrites widenings/narrowings to be explicit in the IR, and
 * attempts to simplify away most of the
 * interleaving/deinterleaving. */
EXPORT Stmt optimize_hexagon_instructions(Stmt s, Target t);

/** Generate deinterleave or interleave operations, operating on
 * groups of vectors at a time. */
//@{
EXPORT Expr native_deinterleave(Expr x);
EXPORT Expr native_interleave(Expr x);
EXPORT bool is_native_deinterleave(Expr x);
EXPORT bool is_native_interleave(Expr x);
//@}

}  // namespace Internal
}  // namespace Halide

#endif
#ifndef HALIDE_INFER_ARGUMENTS_H
#define HALIDE_INFER_ARGUMENTS_H

#include <vector>


/** \file
 *
 * Interface for a visitor to infer arguments used in a body Stmt.
 */

namespace Halide {
namespace Internal {

 /** An inferred argument. Inferred args are either Params,
 * ImageParams, or Buffers. The first two are handled by the param
 * field, and global images are tracked via the buf field. These
 * are used directly when jitting, or used for validation when
 * compiling with an explicit argument list. */
struct InferredArgument {
    Argument arg;
    Parameter param;
    Buffer<> buffer;

    bool operator<(const InferredArgument &other) const {
        if (arg.is_buffer() && !other.arg.is_buffer()) {
            return true;
        } else if (other.arg.is_buffer() && !arg.is_buffer()) {
            return false;
        } else {
            return arg.name < other.arg.name;
        }
    }
};

class Function;

std::vector<InferredArgument> infer_arguments(Stmt body, const std::vector<Function> &outputs);

}  // namespace Internal
}  // namespace Halide

#endif
#ifndef HALIDE_HOST_GPU_BUFFER_COPIES_H
#define HALIDE_HOST_GPU_BUFFER_COPIES_H

/** \file
 * Defines the lowering passes that deal with host and device buffer flow.
 */


namespace Halide {
namespace Internal {

/** A helper function to call an extern function, and assert that it
 * returns 0. */
EXPORT Stmt call_extern_and_assert(const std::string& name, const std::vector<Expr>& args);

/** Inject calls to halide_device_malloc, halide_copy_to_device, and
 * halide_copy_to_host as needed. */
EXPORT Stmt inject_host_dev_buffer_copies(Stmt s, const Target &t);

}
}

#endif
#ifndef HALIDE_INJECT_OPENGL_INTRINSICS_H
#define HALIDE_INJECT_OPENGL_INTRINSICS_H

/** \file
 * Defines the lowering pass that injects texture loads and texture
 * stores for opengl.
 */


namespace Halide {
namespace Internal {

/** Take a statement with for kernel for loops and turn loads and
 * stores inside the loops into OpenGL texture load and store
 * intrinsics. Should only be run when the OpenGL target is active. */
Stmt inject_opengl_intrinsics(Stmt s);

}
}

#endif
#ifndef HALIDE_INLINE_H
#define HALIDE_INLINE_H

/** \file
 * Methods for replacing calls to functions with their definitions.
 */


namespace Halide {
namespace Internal {

/** Inline a single named function, which must be pure. For a pure function to
 * be inlined, it must not have any specializations (i.e. it can only have one
 * values definition). */
// @{
Stmt inline_function(Stmt s, Function f);
Expr inline_function(Expr e, Function f);
void inline_function(Function caller, Function f);
// @}

/** Check if the schedule of an inlined function is legal, throwing an error
 * if it is not. */
void validate_schedule_inlined_function(Function f);

}
}


#endif
#ifndef HALIDE_INLINE_REDUCTIONS_H
#define HALIDE_INLINE_REDUCTIONS_H


/** \file
 * Defines some inline reductions: sum, product, minimum, maximum.
 */
namespace Halide {

/** An inline reduction. This is suitable for convolution-type
 * operations - the reduction will be computed in the innermost loop
 * that it is used in. The argument may contain free or implicit
 * variables, and must refer to some reduction domain. The free
 * variables are still free in the return value, but the reduction
 * domain is captured - the result expression does not refer to a
 * reduction domain and can be used in a pure function definition.
 *
 * An example using \ref sum :
 *
 \code
 Func f, g;
 Var x;
 RDom r(0, 10);
 f(x) = x*x;
 g(x) = sum(f(x + r));
 \endcode
 *
 * Here g computes some blur of x, but g is still a pure function. The
 * sum is being computed by an anonymous reduction function that is
 * scheduled innermost within g.
 */
//@{
EXPORT Expr sum(Expr, const std::string &s = "sum");
EXPORT Expr product(Expr, const std::string &s = "product");
EXPORT Expr maximum(Expr, const std::string &s = "maximum");
EXPORT Expr minimum(Expr, const std::string &s = "minimum");
//@}

/** Variants of the inline reduction in which the RDom is stated
 * explicitly. The expression can refer to multiple RDoms, and only
 * the inner one is captured by the reduction. This allows you to
 * write expressions like:
 \code
 RDom r1(0, 10), r2(0, 10), r3(0, 10);
 Expr e = minimum(r1, product(r2, sum(r3, r1 + r2 + r3)));
 \endcode
*/
// @{
EXPORT Expr sum(RDom, Expr, const std::string &s = "sum");
EXPORT Expr product(RDom, Expr, const std::string &s = "product");
EXPORT Expr maximum(RDom, Expr, const std::string &s = "maximum");
EXPORT Expr minimum(RDom, Expr, const std::string &s = "minimum");
// @}


/** Returns an Expr or Tuple representing the coordinates of the point
 * in the RDom which minimizes or maximizes the expression. The
 * expression must refer to some RDom. Also returns the extreme value
 * of the expression as the last element of the tuple. */
// @{
EXPORT Tuple argmax(Expr, const std::string &s = "argmax");
EXPORT Tuple argmin(Expr, const std::string &s = "argmin");
EXPORT Tuple argmax(RDom, Expr, const std::string &s = "argmax");
EXPORT Tuple argmin(RDom, Expr, const std::string &s = "argmin");
// @}

}

#endif
#ifndef HALIDE_INTEGER_DIVISION_TABLE_H
#define HALIDE_INTEGER_DIVISION_TABLE_H

#include <cstdint>

/** \file
 * Tables telling us how to do integer division via fixed-point
 * multiplication for various small constants. This file is
 * automatically generated by find_inverse.cpp.
 */
namespace Halide {
namespace Internal {
namespace IntegerDivision {
extern const int64_t table_u8[256][4];
extern const int64_t table_s8[256][4];
extern const int64_t table_u16[256][4];
extern const int64_t table_s16[256][4];
extern const int64_t table_u32[256][4];
extern const int64_t table_s32[256][4];
extern const int64_t table_runtime_u8[256][4];
extern const int64_t table_runtime_s8[256][4];
extern const int64_t table_runtime_u16[256][4];
extern const int64_t table_runtime_s16[256][4];
extern const int64_t table_runtime_u32[256][4];
extern const int64_t table_runtime_s32[256][4];
}
}
}

#endif
#ifndef HALIDE_IR_MATCH_H
#define HALIDE_IR_MATCH_H

/** \file
 * Defines a method to match a fragment of IR against a pattern containing wildcards
 */


namespace Halide {
namespace Internal {

/** Does the first expression have the same structure as the second?
 * Variables in the first expression with the name * are interpreted
 * as wildcards, and their matching equivalent in the second
 * expression is placed in the vector give as the third argument.
 * Wildcards require the types to match. For the type bits and width,
 * a 0 indicates "match anything". So an Int(8, 0) will match 8-bit
 * integer vectors of any width (including scalars), and a UInt(0, 0)
 * will match any unsigned integer type.
 *
 * For example:
 \code
 Expr x = Variable::make(Int(32), "*");
 match(x + x, 3 + (2*k), result)
 \endcode
 * should return true, and set result[0] to 3 and
 * result[1] to 2*k.
 */
EXPORT bool expr_match(Expr pattern, Expr expr, std::vector<Expr> &result);

/** Does the first expression have the same structure as the second?
 * Variables are matched consistently. The first time a variable is
 * matched, it assumes the value of the matching part of the second
 * expression. Subsequent matches must be equal to the first match.
 *
 * For example:
 \code
 Var x("x"), y("y");
 match(x*(x + y), a*(a + b), result)
 \endcode
 * should return true, and set result["x"] = a, and result["y"] = b.
 */
EXPORT bool expr_match(Expr pattern, Expr expr, std::map<std::string, Expr> &result);

EXPORT void expr_match_test();

}
}

#endif
#ifndef HALIDE_LAMBDA_H
#define HALIDE_LAMBDA_H


/** \file
 * Convenience functions for creating small anonymous Halide
 * functions. See test/lambda.cpp for example usage. */

namespace Halide {

/** Create a zero-dimensional halide function that returns the given
 * expression. The function may have more dimensions if the expression
 * contains implicit arguments. */
inline Func lambda(Expr e) {
    Func f("lambda" + Internal::unique_name('_'));
    f(_) = e;
    return f;
}

/** Create a 1-D halide function in the first argument that returns
 * the second argument. The function may have more dimensions if the
 * expression contains implicit arguments and the list of Var
 * arguments contains a placeholder ("_"). */
inline Func lambda(Var x, Expr e) {
    Func f("lambda" + Internal::unique_name('_'));
    f(x) = e;
    return f;
}

/** Create a 2-D halide function in the first two arguments that
 * returns the last argument. The function may have more dimensions if
 * the expression contains implicit arguments and the list of Var
 * arguments contains a placeholder ("_"). */
inline Func lambda(Var x, Var y, Expr e) {
    Func f("lambda" + Internal::unique_name('_'));
    f(x, y) = e;
    return f;
}

/** Create a 3-D halide function in the first three arguments that
 * returns the last argument.  The function may have more dimensions
 * if the expression contains implicit arguments and the list of Var
 * arguments contains a placeholder ("_"). */
inline Func lambda(Var x, Var y, Var z, Expr e) {
    Func f("lambda" + Internal::unique_name('_'));
    f(x, y, z) = e;
    return f;
}

/** Create a 4-D halide function in the first four arguments that
 * returns the last argument. The function may have more dimensions if
 * the expression contains implicit arguments and the list of Var
 * arguments contains a placeholder ("_"). */
inline Func lambda(Var x, Var y, Var z, Var w, Expr e) {
    Func f("lambda" + Internal::unique_name('_'));
    f(x, y, z, w) = e;
    return f;
}

/** Create a 5-D halide function in the first five arguments that
 * returns the last argument. The function may have more dimensions if
 * the expression contains implicit arguments and the list of Var
 * arguments contains a placeholder ("_"). */
inline Func lambda(Var x, Var y, Var z, Var w, Var v, Expr e) {
    Func f("lambda" + Internal::unique_name('_'));
    f(x, y, z, w, v) = e;
    return f;
}

}

#endif //HALIDE_LAMBDA_H
#ifndef HALIDE_LERP_H
#define HALIDE_LERP_H

/** \file
 * Defines methods for converting a lerp intrinsic into Halide IR.
 */


namespace Halide {
namespace Internal {

/** Build Halide IR that computes a lerp. Use by codegen targets that
 * don't have a native lerp. */
Expr EXPORT lower_lerp(Expr zero_val, Expr one_val, Expr weight);

}
}

#endif
#ifndef HALIDE_LICM_H
#define HALIDE_LICM_H

/** \file
 * Methods for lifting loop invariants out of inner loops.
 */


namespace Halide {
namespace Internal {

/** Hoist loop-invariants out of inner loops. This is especially
 * important in cases where LLVM would not do it for us
 * automatically. For example, it hoists loop invariants out of cuda
 * kernels. */
Stmt loop_invariant_code_motion(Stmt);

}
}

#endif
#ifndef HALIDE_LLVM_OUTPUTS_H
#define HALIDE_LLVM_OUTPUTS_H

/** \file
 *
 */

#include <string>
#include <vector>


namespace llvm {
class Module;
class TargetOptions;
class LLVMContext;
class raw_fd_ostream;
class raw_pwrite_stream;
class raw_ostream;
}

namespace Halide {

namespace Internal {
typedef llvm::raw_pwrite_stream LLVMOStream;
}

/** Generate an LLVM module. */
EXPORT std::unique_ptr<llvm::Module> compile_module_to_llvm_module(const Module &module, llvm::LLVMContext &context);

/** Construct an llvm output stream for writing to files. */
std::unique_ptr<llvm::raw_fd_ostream> make_raw_fd_ostream(const std::string &filename);

/** Compile an LLVM module to native targets (objects, native assembly). */
// @{
EXPORT void compile_llvm_module_to_object(llvm::Module &module, Internal::LLVMOStream& out);
EXPORT void compile_llvm_module_to_assembly(llvm::Module &module, Internal::LLVMOStream& out);
// @}

/** Compile an LLVM module to LLVM targets (bitcode, LLVM assembly). */
// @{
EXPORT void compile_llvm_module_to_llvm_bitcode(llvm::Module &module, Internal::LLVMOStream& out);
EXPORT void compile_llvm_module_to_llvm_assembly(llvm::Module &module, Internal::LLVMOStream& out);
// @}

/**
 * Concatenate the list of src_files into dst_file, using the appropriate
 * static library format for the given target (e.g., .a or .lib).
 * If deterministic is true, emit 0 for all GID/UID/timestamps, and 0644 for
 * all modes (equivalent to the ar -D option).
 */
EXPORT void create_static_library(const std::vector<std::string> &src_files, const Target &target,
                           const std::string &dst_file, bool deterministic = true);
}

#endif
#ifndef HALIDE_LLVM_RUNTIME_LINKER_H
#define HALIDE_LLVM_RUNTIME_LINKER_H

/** \file
 * Support for linking LLVM modules that comprise the runtime.
 */

#include <memory>

namespace llvm {
class Module;
class LLVMContext;
class Triple;
}  // namespace llvm

namespace Halide {
namespace Internal {

/** Return the llvm::Triple that corresponds to the given Halide Target */
llvm::Triple get_triple_for_target(const Target &target);

/** Create an llvm module containing the support code for a given target. */
std::unique_ptr<llvm::Module> get_initial_module_for_target(Target, llvm::LLVMContext *, bool for_shared_jit_runtime = false, bool just_gpu = false);

/** Create an llvm module containing the support code for ptx device. */
std::unique_ptr<llvm::Module> get_initial_module_for_ptx_device(Target, llvm::LLVMContext *c);

/** Link a block of llvm bitcode into an llvm module. */
void add_bitcode_to_module(llvm::LLVMContext *context, llvm::Module &module,
                           const std::vector<uint8_t> &bitcode, const std::string &name);

}  // namespace Internal
}  // namespace Halide

#endif
#ifndef HALIDE_LOOP_CARRY_H
#define HALIDE_LOOP_CARRY_H


namespace Halide {
namespace Internal {

/** Reuse loads done on previous loop iterations by stashing them in
 * induction variables instead of redoing the load. If the loads are
 * predicated, the predicates need to match. Can be an optimization or
 * pessimization depending on how good the L1 cache is on the architecture
 * and how many memory issue slots there are. Currently only intended
 * for Hexagon. */
Stmt loop_carry(Stmt, int max_carried_values = 8);

}
}

#endif
#ifndef HALIDE_INTERNAL_LOWER_H
#define HALIDE_INTERNAL_LOWER_H

/** \file
 *
 * Defines the function that generates a statement that computes a
 * Halide function using its schedule.
 */

#include <iterator>


namespace Halide {
namespace Internal {

class IRMutator;

/** Given a vector of scheduled halide functions, create a Module that
 * evaluates it. Automatically pulls in all the functions f depends
 * on. Some stages of lowering may be target-specific. The Module may
 * contain submodules for computation offloaded to another execution
 * engine or API as well as buffers that are used in the passed in
 * Stmt. Multiple LoweredFuncs are added to support legacy buffer_t
 * calling convention. */
EXPORT Module lower(const std::vector<Function> &output_funcs, const std::string &pipeline_name, const Target &t,
                    const std::vector<Argument> &args, const Internal::LoweredFunc::LinkageType linkage_type,
                    std::vector<std::string> &order, std::map<std::string, Function> &env,
                    const std::vector<IRMutator *> &custom_passes = std::vector<IRMutator *>(),
                    bool compile_to_tiramisu = false);

/** Given a halide function with a schedule, create a statement that
 * evaluates it. Automatically pulls in all the functions f depends
 * on. Some stages of lowering may be target-specific. Mostly used as
 * a convenience function in tests that wish to assert some property
 * of the lowered IR. */
EXPORT Stmt lower_main_stmt(const std::vector<Function> &output_funcs, const std::string &pipeline_name, const Target &t,
							std::vector<std::string> &order, std::map<std::string, Function> &env,
                            const std::vector<IRMutator *> &custom_passes = std::vector<IRMutator *>(),
                            bool compile_to_tiramisu = false);

void lower_test();

}
}

#endif
/** \file
 * This file only exists to contain the front-page of the documentation
 */

/** \mainpage Halide
 *
 * Halide is a programming language designed to make it easier to
 * write high-performance image processing code on modern
 * machines. Its front end is embedded in C++. Compiler
 * targets include x86/SSE, ARM v7/NEON, CUDA, Native Client,
 * OpenCL, and Metal.
 *
 * You build a Halide program by writing C++ code using objects of
 * type \ref Halide::Var, \ref Halide::Expr, and \ref Halide::Func,
 * and then calling \ref Halide::Func::compile_to_file to generate an
 * object file and header (good for deploying large routines), or
 * calling \ref Halide::Func::realize to JIT-compile and run the
 * pipeline immediately (good for testing small routines).
 *
 * To learn Halide, we recommend you start with the <a href=examples.html>tutorials</a>.
 *
 * You can also look in the test folder for many small examples that
 * use Halide's various features, and in the apps folder for some
 * larger examples that statically compile halide pipelines. In
 * particular check out local_laplacian, bilateral_grid, and
 * interpolate.
 *
 * Below are links to the documentation for the important classes in Halide.
 *
 * For defining, scheduling, and evaluating basic pipelines:
 *
 * Halide::Func, Halide::Stage, Halide::Var
 *
 * Our image data type:
 *
 * Halide::Buffer
 *
 * For passing around and reusing halide expressions:
 *
 * Halide::Expr
 *
 * For representing scalar and image parameters to pipelines:
 *
 * Halide::Param, Halide::ImageParam
 *
 * For writing functions that reduce or scatter over some domain:
 *
 * Halide::RDom
 *
 * For writing and evaluating functions that return multiple values:
 *
 * Halide::Tuple, Halide::Realization
 *
 */

/**
 * \example tutorial/lesson_01_basics.cpp
 * \example tutorial/lesson_02_input_image.cpp
 * \example tutorial/lesson_03_debugging_1.cpp
 * \example tutorial/lesson_04_debugging_2.cpp
 * \example tutorial/lesson_05_scheduling_1.cpp
 * \example tutorial/lesson_06_realizing_over_shifted_domains.cpp
 * \example tutorial/lesson_07_multi_stage_pipelines.cpp
 * \example tutorial/lesson_08_scheduling_2.cpp
 * \example tutorial/lesson_09_update_definitions.cpp
 * \example tutorial/lesson_10_aot_compilation_generate.cpp
 * \example tutorial/lesson_10_aot_compilation_run.cpp
 * \example tutorial/lesson_11_cross_compilation.cpp
 * \example tutorial/lesson_12_using_the_gpu.cpp
 * \example tutorial/lesson_13_tuples.cpp
 * \example tutorial/lesson_14_types.cpp
 * \example tutorial/lesson_15_generators.cpp
 */
#ifndef HALIDE_MATLAB_OUTPUT_H
#define HALIDE_MATLAB_OUTPUT_H

/** \file
 *
 * Provides an output function to generate a Matlab mex API compatible object file.
 */


namespace llvm {
class Module;
class Function;
class Value;
}

namespace Halide {
namespace Internal {

/** Add a mexFunction wrapper definition to the module, calling the
 * function with the name pipeline_name. Returns the mexFunction
 * definition. */
EXPORT llvm::Function *define_matlab_wrapper(llvm::Module *module,
                                             llvm::Function *pipeline_argv_wrapper,
                                             llvm::Function *metadata_getter);

}
}

#endif
#ifndef HALIDE_INTERNAL_CACHING_H
#define HALIDE_INTERNAL_CACHING_H

/** \file
 *
 * Defines the interface to the pass that injects support for
 * compute_cached roots.
 */

#include <map>


namespace Halide {
namespace Internal {

/** Transform pipeline calls for Funcs scheduled with memoize to do a
 *  lookup call to the runtime cache implementation, and if there is a
 *  miss, compute the results and call the runtime to store it back to
 *  the cache.
 *  Should leave non-memoized Funcs unchanged.
 */
Stmt inject_memoization(Stmt s, const std::map<std::string, Function> &env,
                        const std::string &name,
                        const std::vector<Function> &outputs);

/** This should be called after Storage Flattening has added Allocation
 *  IR nodes. It connects the memoization cache lookups to the Allocations
 *  so they point to the buffers from the memoization cache and those buffers
 *  are released when no longer used.
 *  Should not affect allocations for non-memoized Funcs.
 */
Stmt rewrite_memoized_allocations(Stmt s, const std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_MONOTONIC_H
#define HALIDE_MONOTONIC_H

/** \file
 *
 * Methods for computing whether expressions are monotonic
 */


namespace Halide {
namespace Internal {

/**
 * Detect whether an expression is monotonic increasing in a variable,
 * decreasing, or unknown.
 */
enum class Monotonic {Constant, Increasing, Decreasing, Unknown};
EXPORT Monotonic is_monotonic(Expr e, const std::string &var);

EXPORT void is_monotonic_test();

}
}

#endif
#ifndef HALIDE_PARALLEL_RVAR_H
#define HALIDE_PARALLEL_RVAR_H

/** \file
 *
 * Method for checking if it's safe to parallelize an update
 * definition across a reduction variable.
 */


namespace Halide {
namespace Internal {

/** Returns whether or not Halide can prove that it is safe to
 * parallelize an update definition across a specific variable. If
 * this returns true, it's definitely safe. If this returns false, it
 * may still be safe, but Halide couldn't prove it.
 */
bool can_parallelize_rvar(const std::string &rvar,
                          const std::string &func,
                          const Definition &r);

}
}

#endif
#ifndef PARTITION_LOOPS_H
#define PARTITION_LOOPS_H

/** \file
 * Defines a lowering pass that partitions loop bodies into three
 * to handle boundary conditions: A prologue, a simplified
 * steady-stage, and an epilogue.
 */


namespace Halide {
namespace Internal {

/** Return true if an expression uses a likely tag. */
bool has_likely_tag(Expr e);

/** Partitions loop bodies into a prologue, a steady state, and an
 * epilogue. Finds the steady state by hunting for use of clamped
 * ramps, or the 'likely' intrinsic. */
EXPORT Stmt partition_loops(Stmt s);

}
}

#endif
#ifndef HALIDE_PREFETCH_H
#define HALIDE_PREFETCH_H

/** \file
 * Defines the lowering pass that injects prefetch calls when prefetching
 * appears in the schedule.
 */

#include <map>


namespace Halide {
namespace Internal {

Stmt inject_prefetch(Stmt s, const std::map<std::string, Function> &env);

/** Reduce a multi-dimensional prefetch into a prefetch of lower dimension
 * (max dimension of the prefetch is specified by target architecture).
 * This keeps the 'max_dim' innermost dimensions and adds loops for the rest
 * of the dimensions. If maximum prefetched-byte-size is specified (depending
 * on the architecture), this also adds an outer loops that tile the prefetches. */
Stmt reduce_prefetch_dimension(Stmt stmt, const Target &t);

}
}

#endif
#ifndef HALIDE_PROFILING_H
#define HALIDE_PROFILING_H

/** \file
 * Defines the lowering pass that injects print statements when profiling is turned on.
 * The profiler will print out per-pipeline and per-func stats, such as total time
 * spent and heap/stack allocation information. To turn on the profiler, set
 * HL_TARGET/HL_JIT_TARGET flags to 'host-profile'.
 *
 * Output format:
 * \<pipeline_name\>
 *  \<total time spent in this pipeline\> \<# of samples taken\> \<# of runs\> \<avg time/run\>
 *  \<# of heap allocations\> \<peak heap allocation\>
 *   \<func_name\> \<total time spent in this func\> \<percentage of time spent\>
 *     (\<peak heap alloc by this func\> \<num of allocs\> \<average alloc size\> |
 *      \<worst-case peak stack alloc by this func\>)?
 *
 * Sample output:
 * memory_profiler_mandelbrot
 *  total time: 59.832336 ms   samples: 43   runs: 1000   time/run: 0.059832 ms
 *  heap allocations: 104000   peak heap usage: 505344 bytes
 *   f0:          0.025673ms (42%)
 *   mandelbrot:  0.006444ms (10%)   peak: 505344   num: 104000   avg: 5376
 *   argmin:      0.027715ms (46%)   stack: 20
 */


namespace Halide {
namespace Internal {

/** Take a statement representing a halide pipeline insert
 * high-resolution timing into the generated code (via spawning a
 * thread that acts as a sampling profiler); summaries of execution
 * times and counts will be logged at the end. Should be done before
 * storage flattening, but after all bounds inference.
 *
 */
Stmt inject_profiling(Stmt, std::string);

}
}

#endif
#ifndef HALIDE_QUALIFY_H
#define HALIDE_QUALIFY_H

/** \file
 *
 * Defines methods for prefixing names in an expression with a prefix string.
 */


namespace Halide {
namespace Internal {

/** Prefix all variable names in the given expression with the prefix string. */
Expr qualify(const std::string &prefix, Expr value);

}
}


#endif
#ifndef HALIDE_RANDOM_H
#define HALIDE_RANDOM_H

/** \file
 *
 * Defines deterministic random functions, and methods to redirect
 * front-end calls to random_float and random_int to use them. */


namespace Halide {
namespace Internal {

/** Return a random floating-point number between zero and one that
 * varies deterministically based on the input expressions. */
Expr random_float(const std::vector<Expr> &);

/** Return a random unsigned integer between zero and 2^32-1 that
 * varies deterministically based on the input expressions (which must
 * be integers or unsigned integers). */
Expr random_int(const std::vector<Expr> &);

/** Convert calls to random() to IR generated by random_float and
 * random_int. Tags all calls with the variables in free_vars, and the
 * integer given as the last argument. */
Expr lower_random(Expr e, const std::vector<std::string> &free_vars, int tag);

}
}

#endif
#ifndef HALIDE_INTERNAL_REALIZATION_ORDER_H
#define HALIDE_INTERNAL_REALIZATION_ORDER_H

/** \file
 *
 * Defines the lowering pass that determines the order in which
 * realizations are injected.
 */

#include <vector>
#include <string>
#include <map>

namespace Halide {
namespace Internal {

class Function;

/** Given a bunch of functions that call each other, determine an
 * order in which to do the scheduling. This in turn influences the
 * order in which stages are computed when there's no strict
 * dependency between them. Currently just some arbitrary depth-first
 * traversal of the call graph. */
std::vector<std::string> realization_order(const std::vector<Function> &output,
                                           const std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_INTERNAL_REGION_COSTS_H
#define HALIDE_INTERNAL_REGION_COSTS_H

/** \file
 *
 * Defines RegionCosts - used by the auto scheduler to query the cost of
 * computing some function regions.
 */

#include <set>
#include <limits>


namespace Halide {
namespace Internal {

struct Cost {
    // Estimate of cycles spent doing arithmetic.
    Expr arith;
    // Estimate of bytes loaded.
    Expr memory;

    Cost(int64_t arith, int64_t memory) : arith(arith), memory(memory) {}
    Cost(Expr arith, Expr memory) : arith(std::move(arith)), memory(std::move(memory)) {}
    Cost() {}

    inline bool defined() const { return arith.defined() && memory.defined(); }
    void simplify();

    friend std::ostream& operator<<(std::ostream &stream, const Cost &c) {
        stream << "[arith: " << c.arith << ", memory: " << c.memory << "]";
        return stream;
    }
};

/** Auto scheduling component which is used to assign costs for computing a
 * region of a function or one of its stages. */
struct RegionCosts {
    /** An environment map which contains all functions in the pipeline. */
    const std::map<std::string, Function> &env;
    /** A map containing the cost of computing a value in each stage of a
     * function. The number of entries in the vector is equal to the number of
     * stages in the function. */
    std::map<std::string, std::vector<Cost>> func_cost;
    /** A map containing the types of all image inputs in the pipeline. */
    std::map<std::string, Type> inputs;
    /** A scope containing the estimated min/extent values of ImageParams
     * in the pipeline. */
    Scope<Interval> input_estimates;

    /** Return the cost of producing a region (specified by 'bounds') of a
     * function stage (specified by 'func' and 'stage'). 'inlines' specifies
     * names of all the inlined functions. */
    Cost stage_region_cost(std::string func, int stage, const DimBounds &bounds,
                           const std::set<std::string> &inlines = std::set<std::string>());

    /** Return the cost of producing a region of a function stage (specified
     * by 'func' and 'stage'). 'inlines' specifies names of all the inlined
     * functions. */
    Cost stage_region_cost(std::string func, int stage, const Box &region,
                           const std::set<std::string> &inlines = std::set<std::string>());

    /** Return the cost of producing a region of function 'func'. This adds up the
     * costs of all stages of 'func' required to produce the region. 'inlines'
     * specifies names of all the inlined functions. */
    Cost region_cost(std::string func, const Box &region,
                     const std::set<std::string> &inlines = std::set<std::string>());

    /** Same as region_cost above but this computes the total cost of many
     * function regions. */
    Cost region_cost(const std::map<std::string, Box> &regions,
                     const std::set<std::string> &inlines = std::set<std::string>());

    /** Compute the cost of producing a single value by one stage of 'f'.
     * 'inlines' specifies names of all the inlined functions. */
    Cost get_func_stage_cost(const Function &f, int stage,
                             const std::set<std::string> &inlines = std::set<std::string>());

    /** Compute the cost of producing a single value by all stages of 'f'.
     * 'inlines' specifies names of all the inlined functions. This returns a
     * vector of costs. Each entry in the vector corresponds to a stage in 'f'. */
    std::vector<Cost> get_func_cost(const Function &f,
                                    const std::set<std::string> &inlines = std::set<std::string>());

    /** Computes the memory costs of computing a region (specified by 'bounds')
     * of a function stage (specified by 'func' and 'stage'). This returns a map
     * containing the costs incurred to access each of the functions required
     * to produce 'func'. */
    std::map<std::string, Expr>
        stage_detailed_load_costs(std::string func, int stage, DimBounds &bounds,
                                  const std::set<std::string> &inlines = std::set<std::string>());

    /** Return a map containing the costs incurred to access each of the functions
     * required to produce a single value of a function stage. */
    std::map<std::string, Expr>
        stage_detailed_load_costs(std::string func, int stage,
                                  const std::set<std::string> &inlines = std::set<std::string>());

    /** Same as stage_detailed_load_costs above but this computes the cost of a region
     * of 'func'. */
    std::map<std::string, Expr>
        detailed_load_costs(std::string func, const Box &region,
                            const std::set<std::string> &inlines = std::set<std::string>());

    /** Same as detailed_load_costs above but this computes the cost of many function
     * regions and aggregates them. */
    std::map<std::string, Expr>
        detailed_load_costs(const std::map<std::string, Box> &regions,
                            const std::set<std::string> &inlines = std::set<std::string>());

    /** Return the size of the region of 'func' in bytes. */
    Expr region_size(std::string func, const Box &region);

    /** Return the size of the peak amount of memory allocated in bytes. This takes
     * the realization order of the function regions and the early free mechanism
     * into account while computing the peak footprint. */
    Expr region_footprint(const std::map<std::string, Box> &regions,
                          const std::set<std::string> &inlined = std::set<std::string>());

    /** Return the size of the input region in bytes. */
    Expr input_region_size(std::string input, const Box &region);

    /** Return the total size of the many input regions in bytes. */
    Expr input_region_size(const std::map<std::string, Box> &input_regions);

    /** Display the cost of each function in the pipeline. */
    void disp_func_costs();

    /** Construct a region cost object for the pipeline. 'env' is a map of all
     * functions in the pipeline.*/
    RegionCosts(const std::map<std::string, Function> &env);
};

/** Return true if the cost of inlining a function is equivalent to the
 * cost of calling the function directly. */
bool is_func_trivial_to_inline(const Function &func);

}
}

#endif
#ifndef HALIDE_REMOVE_DEAD_ALLOCATIONS_H
#define HALIDE_REMOVE_DEAD_ALLOCATIONS_H

/** \file
 * Defines the lowering pass that removes allocate and free nodes that
 * are not used.
 */


namespace Halide {
namespace Internal {

/** Find Allocate/Free pairs that are never loaded from or stored to,
 *  and remove them from the Stmt. This doesn't touch Realize/Call
 *  nodes and so must be called after storage_flattening.
 */
Stmt remove_dead_allocations(Stmt s);

}
}

#endif
#ifndef HALIDE_REMOVE_TRIVIAL_FOR_LOOPS_H
#define HALIDE_REMOVE_TRIVIAL_FOR_LOOPS_H

/** \file
 * Defines the lowering pass removes for loops of size 1
 */


namespace Halide {
namespace Internal {

/** Convert for loops of size 1 into LetStmt nodes, which allows for
 * further simplification. Done during a late stage of lowering. */
Stmt remove_trivial_for_loops(Stmt s);

}
}

#endif
#ifndef HALIDE_REMOVE_UNDEF
#define HALIDE_REMOVE_UNDEF


/** \file
 * Defines a lowering pass that elides stores that depend on unitialized values.
 */

namespace Halide {
namespace Internal {

/** Removes stores that depend on undef values, and statements that
 * only contain such stores. */
Stmt remove_undef(Stmt s);

}
}

#endif
#ifndef HALIDE_INTERNAL_SCHEDULE_FUNCTIONS_H
#define HALIDE_INTERNAL_SCHEDULE_FUNCTIONS_H

/** \file
 *
 * Defines the function that does initial lowering of Halide Functions
 * into a loop nest using its schedule. The first stage of lowering.
 */

#include <map>


namespace Halide {

struct Target;

namespace Internal {

class Function;

/** Build loop nests and inject Function realizations at the
 * appropriate places using the schedule. Returns a flag indicating
 * whether memoization passes need to be run. */
Stmt schedule_functions(const std::vector<Function> &outputs,
                        const std::vector<std::string> &order,
                        const std::map<std::string, Function> &env,
                        const Target &target,
                        bool compile_to_tiramisu,
                        bool &any_memoized);


}
}

#endif
#ifndef HALIDE_INTERNAL_SELECT_GPU_API_H
#define HALIDE_INTERNAL_SELECT_GPU_API_H


/** \file
 * Defines a lowering pass that selects which GPU api to use for each
 * gpu for loop
 */

namespace Halide {
namespace Internal {

/** Replace for loops with GPU_Default device_api with an actual
 * device API depending on what's enabled in the target. Choose the
 * first of the following: opencl, cuda, openglcompute, opengl */
Stmt select_gpu_api(Stmt s, Target t);

}
}


#endif
#ifndef SIMPLIFY_SPECIALIZATIONS_H
#define SIMPLIFY_SPECIALIZATIONS_H

/** \file
 *
 * Defines pass that try to simplify the RHS/LHS of a function's definition
 * based on its specializations.
 */

#include <map>


namespace Halide {
namespace Internal {

/** Try to simplify the RHS/LHS of a function's definition based on its
 * specializations. */
EXPORT void simplify_specializations(std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_SKIP_STAGES
#define HALIDE_SKIP_STAGES


/** \file
 * Defines a pass that dynamically avoids realizing unnecessary stages.
 */

namespace Halide {
namespace Internal {

/** Avoid computing certain stages if we can infer a runtime condition
 * to check that tells us they won't be used. Does this by analyzing
 * all reads of each buffer allocated, and inferring some condition
 * that tells us if the reads occur. If the condition is non-trivial,
 * inject ifs that guard the production. */
Stmt skip_stages(Stmt s, const std::vector<std::string> &order);

}
}

#endif
#ifndef HALIDE_SLIDING_WINDOW_H
#define HALIDE_SLIDING_WINDOW_H

/** \file
 *
 * Defines the sliding_window lowering optimization pass, which avoids
 * computing provably-already-computed values.
 */

#include <map>


namespace Halide {
namespace Internal {

/** Perform sliding window optimizations on a halide
 * statement. I.e. don't bother computing points in a function that
 * have provably already been computed by a previous iteration.
 */
Stmt sliding_window(Stmt s, const std::map<std::string, Function> &env);

}
}

#endif
#ifndef SOLVE_H
#define SOLVE_H

/** Defines methods for manipulating and analyzing boolean expressions. */


namespace Halide {
namespace Internal {

struct SolverResult {
        Expr result;
        bool fully_solved;
};

/** Attempts to collect all instances of a variable in an expression
 * tree and place it as far to the left as possible, and as far up the
 * tree as possible (i.e. outside most parentheses). If the expression
 * is an equality or comparison, this 'solves' the equation. Returns a
 * pair of Expr and bool. The Expr is the mutated expression, and the
 * bool indicates whether there is a single instance of the variable
 * in the result. If it is false, the expression has only been partially
 * solved, and there are still multiple instances of the variable. */
EXPORT SolverResult solve_expression(
        Expr e, const std::string &variable,
        const Scope<Expr> &scope = Scope<Expr>::empty_scope());

/** Find the smallest interval such that the condition is either true
 * or false inside of it, but definitely false outside of it. Never
 * returns undefined Exprs, instead it uses variables called "pos_inf"
 * and "neg_inf" to represent positive and negative infinity. */
EXPORT Interval solve_for_outer_interval(Expr c, const std::string &variable);

/** Find the largest interval such that the condition is definitely
 * true inside of it, and might be true or false outside of it. */
EXPORT Interval solve_for_inner_interval(Expr c, const std::string &variable);

/** Take a conditional that includes variables that vary over some
 * domain, and convert it to a more conservative (less frequently
 * true) condition that doesn't depend on those variables. Formally,
 * the output expr implies the input expr.
 *
 * The condition may be a vector condition, in which case we also
 * 'and' over the vector lanes, and return a scalar result. */
Expr and_condition_over_domain(Expr c, const Scope<Interval> &varying);

EXPORT void solve_test();

}
}

#endif
#ifndef HALIDE_SPLIT_TUPLES_H
#define HALIDE_SPLIT_TUPLES_H

#include <map>

/** \file
 * Defines the lowering pass that breaks up Tuple-valued realization
 * and productions into several scalar-valued ones. */

namespace Halide {
namespace Internal {


/** Rewrite all tuple-valued Realizations, Provide nodes, and Call
 * nodes into several scalar-valued ones, so that later lowering
 * passes only need to think about scalar-valued productions. */

Stmt split_tuples(Stmt s, const std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_STMT_TO_HTML
#define HALIDE_STMT_TO_HTML

/** \file
 * Defines a function to dump an HTML-formatted stmt to a file.
 */


namespace Halide {
namespace Internal {

/**
 * Dump an HTML-formatted print of a Stmt to filename.
 */
EXPORT void print_to_html(std::string filename, Stmt s);

/** Dump an HTML-formatted print of a Module to filename. */
EXPORT void print_to_html(std::string filename, const Module &m);

}}

#endif
#ifndef HALIDE_STORAGE_FLATTENING_H
#define HALIDE_STORAGE_FLATTENING_H

/** \file
 * Defines the lowering pass that flattens multi-dimensional storage
 * into single-dimensional array access
 */

#include <map>


namespace Halide {
namespace Internal {

/** Take a statement with multi-dimensional Realize, Provide, and Call
 * nodes, and turn it into a statement with single-dimensional
 * Allocate, Store, and Load nodes respectively. */
Stmt storage_flattening(Stmt s,
                        const std::vector<Function> &outputs,
                        const std::map<std::string, Function> &env,
                        const Target &target);

}
}

#endif
#ifndef HALIDE_STORAGE_FOLDING_H
#define HALIDE_STORAGE_FOLDING_H

/** \file
 * Defines the lowering optimization pass that reduces large buffers
 * down to smaller circular buffers when possible
 */


namespace Halide {
namespace Internal {

/** Fold storage of functions if possible. This means reducing one of
 * the dimensions module something for the purpose of storage, if we
 * can prove that this is safe to do. E.g consider:
 *
 \code
 f(x) = ...
 g(x) = f(x-1) + f(x)
 f.store_root().compute_at(g, x);
 \endcode
 *
 * We can store f as a circular buffer of size two, instead of
 * allocating space for all of it.
 */
Stmt storage_folding(Stmt s, const std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_SUBSTITUTE_H
#define HALIDE_SUBSTITUTE_H

/** \file
 *
 * Defines methods for substituting out variables in expressions and
 * statements. */

#include <map>


namespace Halide {
namespace Internal {

/** Substitute variables with the given name with the replacement
 * expression within expr. This is a dangerous thing to do if variable
 * names have not been uniquified. While it won't traverse inside let
 * statements with the same name as the first argument, moving a piece
 * of syntax around can change its meaning, because it can cross lets
 * that redefine variable names that it includes references to. */
EXPORT Expr substitute(const std::string &name, const Expr &replacement, const Expr &expr);

/** Substitute variables with the given name with the replacement
 * expression within stmt. */
EXPORT Stmt substitute(const std::string &name, const Expr &replacement, const Stmt &stmt);

/** Substitute variables with names in the map. */
// @{
EXPORT Expr substitute(const std::map<std::string, Expr> &replacements, const Expr &expr);
EXPORT Stmt substitute(const std::map<std::string, Expr> &replacements, const Stmt &stmt);
// @}

/** Substitute expressions for other expressions. */
// @{
EXPORT Expr substitute(const Expr &find, const Expr &replacement, const Expr &expr);
EXPORT Stmt substitute(const Expr &find, const Expr &replacement, const Stmt &stmt);
// @}

/** Substitutions where the IR may be a general graph (and not just a
 * DAG). */
// @{
Expr graph_substitute(const std::string &name, const Expr &replacement, const Expr &expr);
Stmt graph_substitute(const std::string &name, const Expr &replacement, const Stmt &stmt);
Expr graph_substitute(const Expr &find, const Expr &replacement, const Expr &expr);
Stmt graph_substitute(const Expr &find, const Expr &replacement, const Stmt &stmt);
// @}

/** Substitute in all let Exprs in a piece of IR. Doesn't substitute
 * in let stmts, as this may change the meaning of the IR (e.g. by
 * moving a load after a store). Produces graphs of IR, so don't use
 * non-graph-aware visitors or mutators on it until you've CSE'd the
 * result. */
// @{
Expr substitute_in_all_lets(const Expr &expr);
Stmt substitute_in_all_lets(const Stmt &stmt);
// @}

}
}

#endif
#ifndef HALIDE_THREAD_POOL_H
#define HALIDE_THREAD_POOL_H

#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#ifdef _MSC_VER
#else
#include <unistd.h>
#endif

/** \file
 * Define a simple thread pool utility that is modeled on the api of
 * C++11 std::async(); since implementation details of std::async
 * can vary considerably, with no control over thread spawning, this class
 * allows us to use the same model but with precise control over thread usage.
 *
 * A ThreadPool is created with a specific number of threads, which will never
 * vary over the life of the ThreadPool. (If created without a specific number
 * of threads, it will attempt to use threads == number-of-cores.)
 *
 * Each async request will go into a queue, and will be serviced by the next
 * available thread from the pool.
 *
 * The ThreadPool's dtor will block until all currently-executing tasks
 * to finish (but won't schedule any more).
 *
 * Note that this is a fairly simpleminded ThreadPool, meant for tasks
 * that are fairly coarse (e.g. different tasks in a test); it is specifically
 * *not* intended to be the underlying implementation for Halide runtime threads
 */
namespace Halide {
namespace Internal {

template<typename T>
class ThreadPool {
    struct Job {
        std::function<T()> func;
        std::promise<T> result;

        void run_unlocked(std::unique_lock<std::mutex> &unique_lock);
    };

    // all fields are protected by this mutex.
    std::mutex mutex;

    // Queue of Jobs.
    std::queue<Job> jobs;

    // Broadcast whenever items are added to the Job queue.
    std::condition_variable wakeup_threads;

    // Keep track of threads so they can be joined at shutdown
    std::vector<std::thread> threads;

    // True if the pool is shutting down.
    bool shutting_down{false};

    void worker_thread() {
        std::unique_lock<std::mutex> unique_lock(mutex);
        while (!shutting_down) {
            if (jobs.empty()) {
                // There are no jobs pending. Wait until more jobs are enqueued.
                wakeup_threads.wait(unique_lock);
            } else {
                // Grab the next job.
                Job cur_job = std::move(jobs.front());
                jobs.pop();
                cur_job.run_unlocked(unique_lock);
            }
        }
    }

public:

    static size_t num_processors_online() {
    #ifdef _WIN32
        char *num_cores = getenv("NUMBER_OF_PROCESSORS");
        return num_cores ? atoi(num_cores) : 8;
    #else
        return sysconf(_SC_NPROCESSORS_ONLN);
    #endif
    }

    // Default to number of available cores if not specified otherwise
    ThreadPool(size_t desired_num_threads = num_processors_online()) {
        assert(desired_num_threads > 0);

        std::lock_guard<std::mutex> lock(mutex);

        // Create all the threads.
        for (size_t i = 0; i < desired_num_threads; ++i) {
            threads.emplace_back([this]{ worker_thread(); });
        }
    }

    ~ThreadPool() {
        // Wake everyone up and tell them the party's over and it's time to go home
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutting_down = true;
            wakeup_threads.notify_all();
        }

        // Wait until they leave
        for (auto &t : threads) {
            t.join();
        }
    }

    template<typename Func, typename... Args>
    std::future<T> async(Func func, Args... args) {
        std::lock_guard<std::mutex> lock(mutex);

        Job job;
        // Don't use std::forward here: we never want args passed by reference,
        // since they will be accessed from an arbitrary thread.
        //
        // Some versions of GCC won't allow capturing variadic arguments in a lambda;
        //
        //     job.func = [func, args...]() -> T { return func(args...); };  // Nope, sorry
        //
        // fortunately, we can use std::bind() to accomplish the same thing.
        job.func = std::bind(func, args...);
        jobs.emplace(std::move(job));
        std::future<T> result = jobs.back().result.get_future();

        // Wake up our threads.
        wakeup_threads.notify_all();

        return result;
    }
};

template<typename T>
inline void ThreadPool<T>::Job::run_unlocked(std::unique_lock<std::mutex> &unique_lock) {
    unique_lock.unlock();
    T r = func();
    unique_lock.lock();
    result.set_value(std::move(r));
}

template<>
inline void ThreadPool<void>::Job::run_unlocked(std::unique_lock<std::mutex> &unique_lock) {
    unique_lock.unlock();
    func();
    unique_lock.lock();
    result.set_value();
}


}  // namespace Internal
}  // namespace Halide

#endif  // HALIDE_THREAD_POOL_H
#ifndef HALIDE_TRACING_H
#define HALIDE_TRACING_H

/** \file
 * Defines the lowering pass that injects print statements when tracing is turned on
 */

#include <map>


namespace Halide {
namespace Internal {

/** Take a statement representing a halide pipeline, inject calls to
 * tracing functions at interesting points, such as
 * allocations. Should be done before storage flattening, but after
 * all bounds inference. */
Stmt inject_tracing(Stmt, const std::string &pipeline_name,
                    const std::map<std::string, Function> &env,
                    const std::vector<Function> &outputs,
                    const Target &Target);

}
}

#endif
#ifndef TRIM_NO_OPS_H
#define TRIM_NO_OPS_H

/** \file
 * Defines a lowering pass that truncates loops to the region over
 * which they actually do something.
 */


namespace Halide {
namespace Internal {

/** Truncate loop bounds to the region over which they actually do
 * something. For examples see test/correctness/trim_no_ops.cpp */
EXPORT Stmt trim_no_ops(Stmt s);

}
}

#endif
#ifndef HALIDE_UNIFY_DUPLICATE_LETS_H
#define HALIDE_UNIFY_DUPLICATE_LETS_H

/** \file
 * Defines the lowering pass that coalesces redundant let statements
 */


namespace Halide {
namespace Internal {

/** Find let statements that all define the same value, and make later
 * ones just reuse the symbol names of the earlier ones. */
Stmt unify_duplicate_lets(Stmt s);

}
}

#endif
#ifndef HALIDE_UNIQUIFY_VARIABLE_NAMES
#define HALIDE_UNIQUIFY_VARIABLE_NAMES

/** \file
 * Defines the lowering pass that renames all variables to have unique names.
 */


namespace Halide {
namespace Internal {

/** Modify a statement so that every internally-defined variable name
 * is unique. This lets later passes assume syntactic equivalence is
 * semantic equivalence. */
Stmt uniquify_variable_names(Stmt s);

}
}


#endif
#ifndef HALIDE_UNPACK_BUFFERS_H
#define HALIDE_UNPACK_BUFFERS_H

/** \file
 * Defines the lowering pass that unpacks buffer arguments onto the symbol table
 */


namespace Halide {
namespace Internal {

/** Creates let stmts for the various buffer components
 * (e.g. foo.extent.0) in any referenced concrete buffers or buffer
 * parameters. After this pass, the only undefined symbols should
 * scalar parameters and the buffers themselves (e.g. foo.buffer). */
Stmt unpack_buffers(Stmt s);

}
}

#endif
#ifndef HALIDE_UNROLL_LOOPS_H
#define HALIDE_UNROLL_LOOPS_H

/** \file
 * Defines the lowering pass that unrolls loops marked as such
 */


namespace Halide {
namespace Internal {

/** Take a statement with for loops marked for unrolling, and convert
 * each into several copies of the innermost statement. I.e. unroll
 * the loop. */
Stmt unroll_loops(Stmt);

}
}

#endif
#ifndef __HALIDE_VARYING_ATTRIBUTES__H
#define __HALIDE_VARYING_ATTRIBUTES__H

/** \file
 * This file contains functions that detect expressions in a GLSL scheduled
 * function that may be evaluated per vertex and interpolated across the domain
 * instead of being evaluated at each pixel location across the image.
 */


namespace Halide {
namespace Internal {

/** find_linear_expressions(Stmt s) identifies expressions that may be moved
 * out of the generated fragment shader into a varying attribute. These
 * expressions are tagged by wrapping them in a glsl_varying intrinsic
 */
Stmt find_linear_expressions(Stmt s);

/** Compute a set of 2D mesh coordinates based on the behavior of varying
 * attribute expressions contained within a GLSL scheduled for loop. This
 * method is called during lowering to extract varying attribute
 * expressions and generate code to evalue them at each mesh vertex
 * location. The operation is performed on the host before the draw call
 * to invoke the shader
 */
Stmt setup_gpu_vertex_buffer(Stmt s);

}
}

#endif
#ifndef HALIDE_VECTORIZE_LOOPS_H
#define HALIDE_VECTORIZE_LOOPS_H

/** \file
 * Defines the lowering pass that vectorizes loops marked as such
 */


namespace Halide {
namespace Internal {

/** Take a statement with for loops marked for vectorization, and turn
 * them into single statements that operate on vectors. The loops in
 * question must have constant extent.
 */
Stmt vectorize_loops(Stmt s, const Target &t);

}
}

#endif
#ifndef HALIDE_WRAP_CALLS_H
#define HALIDE_WRAP_CALLS_H

/** \file
 *
 * Defines pass to replace calls to wrapped Functions with their wrappers.
 */

#include <map>


namespace Halide {
namespace Internal {

/** Replace every call to wrapped Functions in the Functions' definitions with
  * call to their wrapper functions. */
std::map<std::string, Function> wrap_func_calls(const std::map<std::string, Function> &env);

}
}

#endif
#ifndef HALIDE_WRAP_EXTERN_STAGES_H
#define HALIDE_WRAP_EXTERN_STAGES_H


/** \file
 *
 * Defines a pass over a Module that adds wrapper LoweredFuncs to any
 * extern stages that need them */

namespace Halide {
namespace Internal {

/** Add wrappers for any LoweredFuncs that need them to support
 * backwards compatibility. This currently wraps extern calls to
 * stages that expect the old buffer_t type. */
void wrap_legacy_extern_stages(Module m);

/** Add a wrapper for a LoweredFunc that accepts old buffers and
 * upgrades them. */
void add_legacy_wrapper(Module m, const LoweredFunc &fn);

}
}

#endif
// This file gets included at the end of Halide.h


// Clean up macros used inside Halide headers
#undef user_assert
#undef user_error
#undef user_warning
#undef internal_error
#undef internal_assert
#undef halide_runtime_error
#undef EXPORT
