all:
	make hello.o hello

hello.o: hello.cu
	nvcc -o hello.o -c hello.cu

hello: hello.c hello.o
	CC -o hello hello.o hello.c

clean:
	rm -f hello.o hello
