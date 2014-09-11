#ifndef PARAMETERS_SCP
#define PARAMETERS_SCP

#include "vector.hpp"

#define NUMTYPE float




typedef NUMTYPE numtype;
typedef cuda_vec<numtype> vec;

namespace parameters
{

  const unsigned int my_N_particle = 7168; 
 // general:
#if __MY_ARCH__ >= 200
  const unsigned int blocksize = 64;                 // blocksize
  const unsigned int min_blockspermp = 8;             // blocks per sm
#else
  const unsigned int blocksize = 128;                 // blocksize
#endif

  const unsigned int N_particle = (my_N_particle/blocksize)*blocksize;     // number of particles  



  // physical constants:
  const double u_mass_SI = 1.660538921e-27; // kg  
  const double charge_e_SI = 1.602176565e-19; // C
  const double epsilon_0_SI = 8.85418781762e-12; // C^2 s^2/(kg m^3)
  const double my_PI = 3.141592654;
  const double boltzmann_SI = 1.3806488e-23; // J/K = m^2/s^2*kg/K

  // UNITS:

  const double delta_t_SI = 1.0e-9;                  // length of time step [s]
  const double Mass = 24;                            // m=M*u
  const double particle_mass_SI = Mass*u_mass_SI;    // mass of particle [kg]
  const int charge = 1;                              // charge factor Q = q/e
  const double particle_charge_SI = charge * charge_e_SI;

  // UNIT convert:
  // f.e. delta_t_SI = 1e-9 sec = 1 u_sec --> chi_time = sec = 1 u_sec/1e-9

  // time : seconds
  // delta_t_SI = 1e-9 sec = 1 u_sec --> chi_time = sec = 1 u_sec/1e-9
  const double chi_time = 1.0/delta_t_SI;
  
  // length : meter
  // 10 mu_meter = 1 u_meter
 
  const double chi_length = 1.0e5; // 3rd root of (harmonic/Coulomb) at end

  // mass : kilogramm
  // u_mass_SI = 1 u_kg --> chi_mass = kg = 1 u_kg/u_mass_SI
  const double chi_mass = 1.0/u_mass_SI;

  // charge : Coulomb
  // 1 q_electron = 1.6e-19 C = 1 u_C
  const double chi_charge = 1.0/charge_e_SI;

  // Temperatur : Kelvin
  // 1 / k_B = 1 u_K
  const double chi_Temp = 1.0;


  // CONVERT for simulation:
  const numtype delta_t = numtype(delta_t_SI * chi_time);
  const numtype particle_mass = numtype(particle_mass_SI*chi_mass);



 
  // finding ground state:
  const unsigned int N_max_iterations = 100000 * N_particle; // maximal iterations for finding the groundstate
  const numtype epsilon_speed_SI = 1.0e-1;//1.0e-4;   // average speed to stop iteration
  const numtype epsilon_speed = numtype(epsilon_speed_SI*chi_length/chi_time);   

  // heating:
  const double boltzmann = boltzmann_SI *chi_length*chi_length/(chi_time*chi_time)*chi_mass/chi_Temp;
  const double Temperatur_SI = 1000000;           // start temperature [K]
                                                 // v^2*m/(2*k_B) = T --> <v>= 26000 m/s
  const double Temperature = Temperatur_SI * chi_Temp; 
  
  // cool down:
  const numtype t_end_SI = 1000*delta_t_SI;                      // end time of cool down simulation
  const numtype t_end = t_end_SI * chi_time;

  // forces:
  // harmonic:
  const double Length_SI_x         = 5.0e-3;            // Length parameter of Paul trap [m]
                                                 // Length between 1 to 50 mm
  const double Length_SI_y         = 5.0e-3;
  const double Length_SI_z         = 5.0e-3;
  const double Voltage_SI        = 1000;              // Voltage at Paul trap [V]
                                                 // Voltage between 100 and 1000 V


  const double harmonic_SI_x = 2.0 * particle_charge_SI * Voltage_SI/(Length_SI_x*Length_SI_x) /particle_mass_SI;
  const double harmonic_SI_y = 2.0 * particle_charge_SI * Voltage_SI/(Length_SI_y*Length_SI_y) /particle_mass_SI;
  const double harmonic_SI_z = 2.0 * particle_charge_SI * Voltage_SI/(Length_SI_z*Length_SI_z) /particle_mass_SI;
  const numtype harmonic_x       = numtype(harmonic_SI_x / (chi_time*chi_time) ) ;
  const numtype harmonic_y       = numtype(harmonic_SI_y / (chi_time*chi_time) ) ;
  const numtype harmonic_z       = numtype(harmonic_SI_z / (chi_time*chi_time) ) ;
                                                 // parameter harmonic force [N/m/kg]
                                                 // = k/m = 2*e*Q * lambda * U/L^2 /m 
                                                 // = 3.204352e-19*Q * lambda * U/L^2 /m (Paul trap)

  // friction:

  const double gamma_SI      = 2.5e7;  // UNITS?
  const double friction_SI   = 2.0*gamma_SI; // UNITS?
  const numtype friction       = numtype(friction_SI/chi_time);
                                                 // parameter friction [N*s/m/kg]
                                                 // = 2*gamma*m/m

  // Coulomb:
  const double coulomb_SI = 1.0/(4.0*my_PI*epsilon_0_SI)*particle_charge_SI*particle_charge_SI/particle_mass_SI;
  const numtype coulomb     =  numtype( coulomb_SI * (1.0/(chi_time*chi_time))*(chi_length*chi_length*chi_length) ); 
                                                 // parameter Coulomb force [N*m^2/kg]
                                                 // = e^2*Q^2/(4 pi epsilon_0 m) = 2.3070755568e-28 * Q^2/m

  const numtype epsilon_yukawa = 1.0e-10f*chi_length;     // protecting agains divergence


}




#endif
