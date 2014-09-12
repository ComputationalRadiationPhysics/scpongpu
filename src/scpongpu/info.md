Main source code of SCPonGPU
========

In `scpongpu/src/scpongpu`, the entire source code of SCPonGPU is located.

If you want to run SCPonGPU, go to that directory.

You need to load a variety of libraries:

 - gcc (works with gcc/4.6.2)
 - nvcc (works with cuda/5.5)


On hypnos, this can be done by running the following commands on a GPU node

```bash
$ source scpongpu.profile
$ make
$ ./scp.exe
```
