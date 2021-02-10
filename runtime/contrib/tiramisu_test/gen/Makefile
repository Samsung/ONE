# need to set TIRAMISU path

all: arithmetic_gen arithmetic_run

arithmetic_gen: arithmetic_gen.cpp
	g++ -I${TIRAMISU}/include -I${TIRAMISU}/3rdParty/isl/build/include -I${TIRAMISU}/3rdParty/Halide/include -I${TIRAMISU}/3rdParty/Halide/tools -I${TIRAMISU}/tiramisu/build -std=c++11 -Wall -Wno-sign-compare -fno-rtti -DHALIDE_NO_JPEG -g -O0 -o arithmetic_gen.o -c arithmetic_gen.cpp
	g++ -std=c++11 -Wall -Wno-sign-compare -fno-rtti -DHALIDE_NO_JPEG -g -O0 arithmetic_gen.o -o arithmetic_gen -Wl,-rpath,${TIRAMISU}/build:${TIRAMISU}/3rdParty/isl/build/lib ${TIRAMISU}/build/libtiramisu.so ${TIRAMISU}/3rdParty/Halide/lib/libHalide.a ${TIRAMISU}/3rdParty/isl/build/lib/libisl.so -ldl -lpthread -lrt -lz -lm

arithmetic.o: arithmetic_gen
	./arithmetic_gen

clean:
	rm -f arithmetic.o arithmetic.o.h arithmetic_run arithmetic_gen arithmetic_gen.o
