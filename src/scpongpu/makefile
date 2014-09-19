# make the strong coupled plasma CUDA code

# compiler stuff
CC = nvcc
CFLAGS = -Xptxas -v 

# specify architecture
SM13 = -arch=sm_13 -D__MY_ARCH__=130
SM20 = -arch=sm_20 -D__MY_ARCH__=200
SM35 = -arch=sm_35 -D__MY_ARCH__=350

# flag for fast math (does not exist on sm13 - there  always fast math)
FAST = -ftz=true -prec-div=false -prec-sqrt=false -use_fast_math


# if no specific compilation is selected use fast option
all: sm35fast

# TODO: there is new NVIDIA hardware avaiable - adjust compilation
# issue #5

# best for hardware architectur SM_13
sm13: plasma.kernel plasma.hpp vector.hpp random/* main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM13) main.cu -o scp.exe

# best for hardware architectur SM_20 + no fast math
sm20slow: plasma.kernel plasma.hpp vector.hpp random/* main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM20) main.cu -o scp.exe

# best for hardware architectur SM_20 + with fast math
sm20fast: plasma.kernel plasma.hpp vector.hpp random/* main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM20) $(FAST) main.cu -o scp.exe

# best for hardware architectur SM_35 + no fast math
sm35slow: plasma.kernel plasma.hpp vector.hpp random/* main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM35) main.cu -o scp.exe

# best for hardware architectur SM_35 + with fast math
sm35fast: plasma.kernel plasma.hpp vector.hpp random/* main.cu parameters.hpp utilities.hpp makefile
	$(CC) $(CFLAGS) $(SM35) $(FAST) main.cu -o scp.exe


# remove the executable
clean:
	rm -f *.exe
