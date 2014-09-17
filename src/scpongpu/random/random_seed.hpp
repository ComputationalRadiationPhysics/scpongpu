#include <time.h>
#include <cstdlib>
#include <limits.h>
#include <cmath>
#include <iostream>


#pragma once

/**
 * class to set the same seed once for all random calls
 * use fixed seed to reproduce physics
 */
class Monte_Carlo_Switch_fixed
{
public:
  Monte_Carlo_Switch_fixed()  
    {
      const unsigned int seed = 12;
      srand( seed );  /* set seed once */
      std::cout << " seed was set to: " << seed 
                << " (fixed value)" << std::endl;
    } 
};


/**
 * class to set the seed randomly once for all random calls
 * no reproduction of exact results
 */
class Monte_Carlo_Switch_random
{
public:
  Monte_Carlo_Switch_random()  
    {
      const unsigned int seed = time(NULL)*::getpid();
      srand( seed );  /* set seed once */
      std::cout << " seed was set to: " << seed 
                << " (randomly set)" << std::endl;
    } 
};






