#include <iostream>
#include <cstdio>
#include <cmath>
#include "random/random_boltzmann.hpp"
#include "vector.hpp"
#include "plasma.kernel"

#pragma once

/** 
 * print particle position and velocity to stdout 
*/
void print_particles(vec* location, 
                     vec* speed, 
                     unsigned int N_particle);

/**
 * save particle location and speed to logfile
 */
void save_particles(vec* location, 
                    vec* speed, 
                    unsigned int N_particle, 
                    FILE* logfile);

/**
 * perform a single velocity-verlet step to propagate speed and position for one time step
 */
void inline velocity_verlet_step(vec* location_d, 
                                 vec* speed_d, 
                                 vec* accel_d, 
                                 const unsigned int N_particle,
                                 const numtype delta_t, 
                                 const dim3 gridDim, 
                                 const dim3 blockDim, 
                                 unsigned int i);



/**
 * cool down particles till stability is reached
 *
 * particles are initialized with a physically not correct position
 * by applying the Coulomb force and and a friction force, a physically 
 * correct, cold state is reached
 */
void finding_ground_state(vec* location_d, 
                          vec* speed_d, 
                          vec* accel_d,
                          vec* speed_h,
                          const unsigned int size, 
                          const dim3 gridDim, 
                          const dim3 blockDim)
{
  using namespace parameters;

  /* finding ground state by repeating particle pushes till stability is reached: */
  for(unsigned i=0; i<N_max_iterations; ++i)
    {
      /* one step using velocity-verlet as integrator: */
      velocity_verlet_step(location_d, speed_d, accel_d, N_particle, delta_t, gridDim, blockDim, i);

      /* determine if iteration can be stopped (no changes in particle movement anymore) */
      if(i == N_max_iterations - 1) /* loop counter stop due to max iterations */
        {
          std::cout << std::endl << " iteration stopped: loop counter reached limit of " 
                    << N_max_iterations << std::endl;
        }
      else if(i%1000 == 999) /* in between checks */
        {
          /* verbose output to see that simulation is running / checks are performed */
          std::cout << "*" << std::flush;

          /* check every 1000th step if changes are negligible */
          /* TODO 1000 is a magic number */
          /* copy data from device to host */
          cudaMemcpy(speed_h,    speed_d,    size, cudaMemcpyDeviceToHost);
          /* TODO: planning to implement this in between velocity_verlet steps with asynchronous memcopy */

          numtype total_speed = 0.;
          /* add all speed magnitudes */
          for(unsigned p=0; p< N_particle; ++p) 
            {
              total_speed += std::sqrt(util::square(speed_h[p].x) + 
                                       util::square(speed_h[p].y) + 
                                       util::square(speed_h[p].z));
            }

          /* check if average speed is below limit */
          if(total_speed/N_particle < epsilon_speed) 
            {
              /* verbose output: inform about number of iterations */
              std::cout << std::endl << "   iteration stopped: needed " << i+1 
                        << " iterations" << std::endl << std::endl;  
              break; /* stop iteration */
            }	   
        }
    } /* end for loop (velocity-verlet-steps)*/
}


/**
 * give particles a temperature by overwriting their velocities
 * with Maxwell-Boltzmann distributed speeds
 */
void heating(vec* speed_h, 
             vec* speed_d, 
             const unsigned int size)
{
  using namespace parameters;

  /* instantaneous heat bath: */
  Maxwell_Boltzmann<numtype, SeedSelected> give_speed(Temperature, particle_mass);
  for(unsigned i=0; i<N_particle; ++i)
    {
      speed_h[i].x = give_speed.get(); 
      speed_h[i].y = give_speed.get(); 
      speed_h[i].z = give_speed.get(); 
    }

  /* copy new speed to GPU: */
  cudaMemcpy(speed_d,    speed_h,    size, cudaMemcpyHostToDevice);
  
}


/**
 * simulate laser cooling for predefined time period
 * TODO: should t_end and delta_t be a parameter to this function?
 */
void cool_down(vec* location_d, 
               vec* speed_d, 
               vec* accel_d, 
               const dim3 gridDim, 
               const dim3 blockDim)
{
  using namespace parameters;

  /* cool particles down again: */
  unsigned int i=0; /* step counter */

  /* run from t=0 to t=t_end */
  for(numtype t=0; t<t_end; t+=delta_t) 
    {
      velocity_verlet_step(location_d, speed_d, accel_d, N_particle, delta_t, gridDim, blockDim, i);
      ++i; /* increase step counter */
    }
}


/**
 * print particle position and speed to stdout
 */
void print_particles(vec* location, 
                     vec* speed, 
                     unsigned int N_particle)
{
  /* determine conversion factors from simulation units to SI units */
  using namespace parameters;
  const numtype lengthfactor = 1.0/parameters::chi_length;
  const numtype speedfactor = parameters::chi_time/parameters::chi_length;

  /* print particles with id 0 to N_particles */
  for(unsigned int i=0; i<N_particle; ++i)
    {
      std::cout << "particle: " << i 
                << " \t location: ( " << location[i].x*lengthfactor << " , " 
                                      << location[i].y*lengthfactor << " , " 
                                      << location[i].z*lengthfactor << " ) "
                << " \t speed: ( "  << speed[i].x*speedfactor << " , " 
                                    << speed[i].y*speedfactor << " , " 
                                    << speed[i].z*speedfactor << " ) " 
                << std::endl;
    }
  std::cout << std::endl;
}


/**
 * save particle data to file
 */
void save_particles(vec* location, 
                    vec* speed, 
                    unsigned int N_particle, 
                    FILE* logfile)
{
  /* determine conversion factors from simulation units to SI units */
  using namespace parameters;
  const float lengthfactor = 1.0/parameters::chi_length;
  const float speedfactor = parameters::chi_time/parameters::chi_length;

  /* print particles with id 0 to N_particles to file */
  for(unsigned int i=0; i<N_particle; ++i)
    {
      fprintf(logfile,
              "particle: %u \t location: ( %f , %f , %f )  \t speed: ( %f , %f , %f )\n",
              i,
              location[i].x*lengthfactor, location[i].y*lengthfactor, location[i].z*lengthfactor,
              speed[i].x*speedfactor, speed[i].y*speedfactor, speed[i].z*speedfactor
              );
    }
}


/**
 * perform one velocity-verlet step
 */
void inline velocity_verlet_step(vec* location_d, 
                                 vec* speed_d, 
                                 vec* accel_d, 
                                 const unsigned int N_particle,
                                 const numtype delta_t, 
                                 const dim3 gridDim, 
                                 const dim3 blockDim, 
                                 const unsigned int i)
{
  /* velocity verlet: 1 time step = 2 parts (step_1 and step_2) */

  /* initial first Coulomb force at step 0 */
  if(i==0) 
    {
      next_accel<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
    }

  /* then always do: */
  next_step_1<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
  next_accel<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
  next_step_2<<<gridDim, blockDim>>>(location_d, speed_d, accel_d, N_particle, delta_t);
}




