HALIDE=~/Halide

snap run cmake -DCMAKE_BUILD_TYPE=Debug  ../infra/nncc/
make -j10 circle_codegen
LD_LIBRARY_PATH=${HALIDE}/install/lib/ ./compiler/luci-codegen/src/circle_codegen arithmetic.circle generated

~/ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ -O3 --std=c++17 ../compiler/luci-codegen/src/simple_runner.cpp generated_subgraph_0.a -I ${HALIDE}/install/include/ -ldl -llog -o runner

adb push runner /data/local/tmp
adb shell LD_LIBRARY_PATH=/data/local/tmp taskset -a 32 /data/local/tmp/runner

#g++ -O1 --std=c++17 ../compiler/luci-codegen/src/simple_runner.cpp generated_subgraph_0.o ${HALIDE}/install/lib/libHalide.so -I ${HALIDE}/install/include/ -ldl -lpthread -o runner
#g++ -O0 --std=c++17  ../compiler/luci-codegen/src/simple_runner.cpp generated_subgraph_0.c /home/aefimov/Halide/build/src/runtime/runtime.o /home/aefimov/Halide/build/src/runtime/runtime_msan.o ${HALIDE}/install/lib/libHalide.so -I ${HALIDE}/install/include/ -ldl -lpthread -o runner -g

#LD_LIBRARY_PATH=${HALIDE}/install/lib/ ./runner
