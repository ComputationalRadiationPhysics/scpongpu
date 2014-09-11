# make the sstrong coupled plasma CUDA code

CC = nvcc
CFLAGS = -Xptxas -v 
SM13 = -arch=sm_13 -D__MY_ARCH__=130
SM20 = -arch=sm_20 -D__MY_ARCH__=200
FAST = -ftz=true -prec-div=false -prec-sqrt=false -use_fast_math


all: fast

13: plasma.kernel plasma.hpp vector.hpp monte_carlo.hpp main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM13) main.cu -o scp.exe

slow: plasma.kernel plasma.hpp vector.hpp monte_carlo.hpp main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM20) main.cu -o scp.exe

fast: plasma.kernel plasma.hpp vector.hpp monte_carlo.hpp main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM20) $(FAST) main.cu -o scp.exe

clean:
	rm -f *.exe
