SCPonGPU
========
**Simulation of strongly coupled plasma using GPUs**

SCPonGPU is a code to simulate [laser cooling](http://en.wikipedia.org/wiki/Laser_cooling) of 
ions in storage rings and particle traps. 
In order to simulate the strong coupling of ions in these scenarios, the 
[Coulomb interaction](http://en.wikipedia.org/wiki/Coulomb%27s_law) of all particles 
with each other is considered. 
This approach is called a [molecular dynamics simulation](http://en.wikipedia.org/wiki/Molecular_dynamics), 
short MD, and has a computational complexity of `NxN` (with `N` being the number of particles).
Since this product can be quite large,
we simulate the particle-particle interaction on [GPUs](http://en.wikipedia.org/wiki/Graphics_processing_unit) 
using [CUDA](http://en.wikipedia.org/wiki/CUDA), 
where we can exploit a highly parallel architecture to accelerate the simulation.  



