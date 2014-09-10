SCPonGPU
========
**Simulation of strongly coupled plasma using GPUs**

SCPonGPU is a simulation code to  simulate the laser-ion-cooling process in 
storage rings and particle traps. In order to simulate the strong coupling 
of ions in these scenarios, the interaction of all particles with each other  
is considered. Since this is evaluation with complexity NxN we simulate
the particle-particle interaction on GPUs where we can exploit the highly 
parallel architecture to accelerate the simulation.  



