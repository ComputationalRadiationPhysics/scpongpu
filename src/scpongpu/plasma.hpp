#include <iostream>
#include <cstdio>
#include <cmath>
#include "monte_carlo.hpp"
#include "vector.hpp"
#include "plasma.kernel"

#ifndef PLASMA_HPP
#define PLASMA_HPP

void print_particles(vec* location, vec* speed, unsigned int N_particle);
void save_particles(vec* location, vec* speed, unsigned int N_particle, FILE* logfile);
void inline velocity_verlet_step(vec* location_d, vec* speed_d, vec* accel_d, const unsigned int N_particle,
				 const numtype delta_t, const dim3 gridDim, const dim3 blockDim, unsigned int i);




void finding_ground_state(vec* location_h, vec* speed_h, vec* location_d, vec* speed_d, vec* accel_d,
			  const unsigned int size, const dim3 gridDim, const dim3 blockDim)
{
  using namespace parameters;

  //open Logfile
  //FILE *logfile;
  //logfile = fopen("logfile_groundstate.txt","w");

  // finding ground state:
  for(unsigned i=0; i<N_max_iterations; ++i)
    {
      // one step using velocity-verlet as integrator:
      velocity_verlet_step(location_d, speed_d, accel_d, N_particle, delta_t, gridDim, blockDim, i);

      // determin if iteration can be stopped (no changes in particle movement anymore)
      if(i == N_max_iterations - 1) // loop counter stop
	{
	  std::cout << std::endl << " iteration stopped: loop counter reached limit of " 
		    << N_max_iterations << std::endl;
	}
      else if(i%1000 == 999) // in between checks
	{
	  std::cout << "*" << std::flush;
	  //check every 1000th step if changes are neglectable
	  
	  cudaMemcpy(location_h, location_d, size, cudaMemcpyDeviceToHost); 
	  cudaMemcpy(speed_h,    speed_d,    size, cudaMemcpyDeviceToHost);
	  //save particles to file
	  //this is just the dump implementation
	  //planning to implement this in between velocity_verlet steps with asynchronous memcopy
	  //save_particles(location_h, speed_h, N_particle, logfile);
	  numtype total_speed = 0;
	  for(unsigned p=0; p< N_particle; ++p) // add all speed magnitudes
	    total_speed += std::sqrt(util::square(speed_h[p].x) + 
				     util::square(speed_h[p].y) + 
				     util::square(speed_h[p].z));
	  if(total_speed/N_particle < epsilon_speed) // check if average speed is below limit
	    {
	      std::cout << std::endl << "   iteration stopped: needed " << i+1 
			<< " iterations" << std::endl << std::endl; // inform about number of iterations
	      break; // stop iteration
	    }	   
	}
    } // end for loop
}


void heating(vec* location_h, vec* speed_h, vec* speed_d, const unsigned int size)
{
  using namespace parameters;

  // heat bath:
  Maxwell_Boltzmann<numtype> give_speed(Temperature, particle_mass);
  for(unsigned i=0; i<N_particle; ++i)
    {
      speed_h[i].x = give_speed.get(); // give a particle a Maxwell-Boltzmann
      speed_h[i].y = give_speed.get(); // distributed speed
      speed_h[i].z = give_speed.get(); // (binning plot for illustration?)
    }
  // copy new speed to GPU:
  cudaMemcpy(speed_d,    speed_h,    size, cudaMemcpyHostToDevice);
  // print starting values:
  std::cout << "Start values: " << std::endl; 
  print_particles(location_h, speed_h, N_particle);

}

void cool_down(vec* location_d, vec* speed_d, vec* accel_d, const dim3 gridDim, const dim3 blockDim)
{
  using namespace parameters;

  // cool down again:
  unsigned int i=0;
  for(numtype t=0; t<t_end; t+=delta_t) // run from t=0 to t=t_end 
    {
      velocity_verlet_step(location_d, speed_d, accel_d, N_particle, delta_t, gridDim, blockDim, i);
      ++i;
    }
}





void print_particles(vec* location, vec* speed, unsigned int N_particle)
{
  // print location and speed of particle 0 to N_particle
  using namespace parameters;
  const numtype lengthfactor = 1.0/parameters::chi_length;
  const numtype speedfactor = parameters::chi_time/parameters::chi_length;


  for(unsigned int i=0; i<N_particle; ++i)
    {
      std::cout << "particle: " << i << " \t location: ( " << location[i].x*lengthfactor << " , " 
		<< location[i].y*lengthfactor << " , " << location[i].z*lengthfactor << " )       \t speed: ( "  
		<< speed[i].x*speedfactor << " , " << speed[i].y*speedfactor << " , " << speed[i].z*speedfactor 
		<< " ) " << std::endl;
     }
  std::cout << std::endl;
}


void save_particles(vec* location, vec* speed, unsigned int N_particle, FILE* logfile)
{
  // save location and speed of particle 0 to N_particle
  using namespace parameters;
  const float lengthfactor = 1.0/parameters::chi_length;
  const float speedfactor = parameters::chi_time/parameters::chi_length;


  for(unsigned int i=0; i<N_particle; ++i)
    {
      fprintf(logfile,"particle: %u \t location: ( %f , %f , %f )      \t speed: ( %f , %f , %f )\n",
	      i,location[i].x*lengthfactor,location[i].y*lengthfactor,location[i].z*lengthfactor,
	      speed[i].x*speedfactor,speed[i].y*speedfactor,speed[i].z*speedfactor);
     }
}



void inline velocity_verlet_step(vec* location_d, vec* speed_d, vec* accel_d, const unsigned int N_particle,
				 const numtype delta_t, const dim3 gridDim, const dim3 blockDim, const unsigned int i)
{
  // velocity verlet: 2 parts = 1 step
  if(i==0) // inital first Coulomb force
    next_accel<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);

  next_step_1<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
  next_accel<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
  next_step_2<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
}



#endif
