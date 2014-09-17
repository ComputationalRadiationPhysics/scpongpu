Development goals for SCPonGPU
========

SCPonGPU should be used to simulate the laser colling of ions in storage rings and traps.
The short term goals are to add the following features:

**Particle simulation code**

 - add *time dependent* external potential for trapping the ions
 - add laser-cooling force based on a *Monte-Carlo* simulation of exciting/deexciting the ions
 - add *inter-beam-scattering* (IBS) force based on a Monte-Carlo approach (rest gas collides with ions)
 - add *bunching potential* V(t,s)
 - restructure *kernel calls* for forces, particle pushes, analysis

Additionally, a diagnostic system could be added:

**Virtual diagnostics**

 - add code to get the  Schottky noise spectrum
 - simulate emitted radiation from the ion transitions as seen on a detector


 