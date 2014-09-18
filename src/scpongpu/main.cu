#include "plasma.hpp"


int main(void)
{
  /* parameters are in namespace parameters */
  using namespace parameters;

  /*############################################################*/
  /*               prepare simulation                           */
  /*############################################################*/

  /* verbose output of particles and parallelization */
  std::cout << "Number of particles: " << N_particle << std::endl;
  std::cout << "blocksize:           " << blocksize << std::endl;

  /* set up start parameters and copy them to GPU:*/
  vec location_h[N_particle];   /* particle location on host */
  vec speed_h[N_particle];      /* particle speed on host*/
  vec accel_h[N_particle];      /* particle acceleration on host*/

  /* TODO replace magic numbers */
  Uniformly<numtype, SeedSelected> uni(-1.0e-5f, 1.0e-5f);  /* random distribution for positioning */
  for(unsigned int i=0; i<N_particle; ++i)
    {
      /* give each particle a random position */
      location_h[i].x = uni.get() * chi_length;
      location_h[i].y = uni.get() * chi_length;
      location_h[i].z = uni.get() * chi_length;
      
      /* give each particle a zeros speed and zero acceleration */      
      speed_h[i].zero();
      accel_h[i].zero();
        
    }

  /* print iteration starting values: */
  std::cout << " Start values of finding the ground state: " << std::endl; 
  print_particles(location_h, speed_h, N_particle);

  /* prepare GPU part: */
  unsigned int size = sizeof(vec) * N_particle; /* size of one copy of all particles */

  /* device pointers: */
  vec *location_d; 
  vec *speed_d;
  vec *accel_d;

  /* allocate and copy position, speed, acceleration */
  cudaMalloc((void**)&location_d, size);
  cudaMalloc((void**)&speed_d,    size);
  cudaMalloc((void**)&accel_d,    size);
  cudaMemcpy(location_d, location_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(speed_d,    speed_h,    size, cudaMemcpyHostToDevice);
  cudaMemcpy(accel_d,    accel_h,    size, cudaMemcpyHostToDevice);
 
  /* set parallelization parameters: */
  dim3 gridDim(N_particle/blocksize, 1, 1);
  dim3 blockDim(blocksize, 1, 1);


  /* ################################################################ */
  /*                     run simulation                               */
  /* ################################################################ */

  /* find physically correct start position */
  finding_ground_state(location_d, speed_d, accel_d, speed_h, size, gridDim, blockDim);
  cudaMemcpy(location_h, location_d, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(speed_h,    speed_d,    size, cudaMemcpyDeviceToHost);

  /* overwrite particle speed to set predefined temperature */
  heating(speed_h, speed_d, size);
  /* print starting values: */
  /* verbose output */
  std::cout << "Start values: " << std::endl; 
  print_particles(location_h, speed_h, N_particle);

  /* simulate laser cooling */
  cool_down(location_d, speed_d, accel_d, gridDim, blockDim);





  /* ################################################################# */
  /*                finish simulation                                  */
  /* ################################################################# */

  /* copy results from GPU to CPU: */
  cudaMemcpy(location_h, location_d, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(speed_h,    speed_d,    size, cudaMemcpyDeviceToHost);
  cudaMemcpy(accel_h,    accel_d,    size, cudaMemcpyDeviceToHost);
  cudaFree(location_d);
  cudaFree(speed_d);
  cudaFree(accel_d);

  /* print results: */
  std::cout << std::endl << std::endl << "final values: " << std::endl;
  print_particles(location_h, speed_h, N_particle);

  return 0;
}



