#include "plasma.hpp"





int main(void)
{
  // set parameters:
  using namespace parameters;

  // ############################################################

  std::cout << "Number of particles: " << N_particle << std::endl;
  std::cout << "blocksize:           " << blocksize << std::endl;

  // set up start parameters and copy them to GPU:
  vec location_h[N_particle];   // particle loaction
  vec speed_h[N_particle];      // particle speed

  Uniformly<numtype> uni(-1.0e-5f, 1.0e-5f);  // random distribution for positioning
  for(unsigned int i=0; i<N_particle; ++i)
    {
      location_h[i].x = uni.get() * chi_length;
      location_h[i].y = uni.get() * chi_length;
      location_h[i].z = uni.get() * chi_length;
      //speed_h[i].x = uni.get() * chi_length/chi_time;
      //speed_h[i].y = uni.get() * chi_length/chi_time;
      //speed_h[i].z = uni.get() * chi_length/chi_time;
    }

  // print iteration starting values:
  std::cout << " Start values of finding the ground state: " << std::endl; 
  //print_particles(location_h, speed_h, N_particle);

  // prepare GPU part
  unsigned int size = sizeof(vec) * N_particle;
  vec *location_d;
  vec *speed_d;
  vec *accel_d;
  cudaMalloc((void**)&location_d, size);
  cudaMalloc((void**)&speed_d,    size);
  cudaMalloc((void**)&accel_d,    size);
  cudaMemcpy(location_d, location_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(speed_d,    speed_h,    size, cudaMemcpyHostToDevice);
 
  dim3 gridDim(N_particle/blocksize, 1, 1);
  dim3 blockDim(blocksize, 1, 1);




  // ################################################################

  // run simulation:
  finding_ground_state(location_h, speed_h, location_d, speed_d, accel_d, size, gridDim, blockDim);
  heating(location_h, speed_h, speed_d, size);
  cool_down(location_d, speed_d, accel_d, gridDim, blockDim);

  // #################################################################




  // copy results to CPU:
  cudaMemcpy(location_h, location_d, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(speed_h,    speed_d,    size, cudaMemcpyDeviceToHost);
  cudaFree(location_d);
  cudaFree(speed_d);

  // print results:
  std::cout << std::endl << std::endl << "final values: " << std::endl;
  print_particles(location_h, speed_h, N_particle);

  return 0;
}



