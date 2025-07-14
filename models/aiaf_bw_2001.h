/*
 *  iaf_bw_2001_w_k.cpp
 *
 *  This file implements the iaf_bw_2001_w_k neuron model.
 *
 *  Author: OpenAI's GPT-4, based on the NEST iaf_bw_2001 model.
 */

#include "iaf_bw_2001_w_k.h"
#include "exceptions.h"
#include "numerics.h"
#include "universal_data_logger_impl.h"

#include <cmath>

namespace nest
{
// The register_neuron macro is now in a separate file that is automatically
// included by the build system.

void
iaf_bw_2001_w_k::calibrate( )
{
  // This function is called once at the beginning of each simulation.
  // We use it to calculate frequently used values.

  // Check that parameters are valid
  if ( P_.t_ref < Time::get_resolution( ).get_ms( ) )
    throw BadProperty( "Refractory period cannot be smaller than the simulation resolution." );
  if ( P_.tau_m <= 0.0 )
    throw BadProperty( "Membrane time constant must be positive." );
  if ( P_.C_m <= 0.0 )
    throw BadProperty( "Membrane capacitance must be positive." );
  if ( P_.tau_w <= 0.0 )
    throw BadProperty( "Adaptation time constant tau_w must be positive." );


  const double h = Time::get_resolution( ).get_ms( );

  // Pre-calculate the exponential decay factors for V_m and w
  S_.exp_m_ = std::exp( -h / P_.tau_m );
  S_.exp_w_ = std::exp( -h / P_.tau_w );

  // Pre-calculate the integration constant C1
  // This is derived from the exact integration of the membrane equation:
  // V(t+h) = V_rest + (V(t)-V_rest)*exp(-h/tau_m) + (R_m*I_total)*(1-exp(-h/tau_m))
  // where R_m = tau_m / C_m.
  // C1 = R_m * (1 - exp_m) = (tau_m / C_m) * (1 - exp_m)
  S_.C1_ = ( P_.tau_m / P_.C_m ) * ( 1.0 - S_.exp_m_ );

  // Set initial refractory time
  S_.ref_end_ = 0;
}

void
iaf_bw_2001_w_k::update( const Time& t, const std::vector< double >& I, const std::vector< double >& )
{
  // Get the synaptic current for this time step, which is the first element
  // of the input vector I.
  const double I_syn = I[ 0 ];

  // If the neuron is in its refractory period, do nothing but update the
  // refractory counter. The membrane potential is clamped to V_reset.
  if ( t.get_steps( ) < S_.ref_end_ )
  {
    S_.V_m = P_.V_reset;
    return;
  }

  // Update adaptation current w: w(t+h) = w(t) * exp(-h/tau_w)
  S_.w *= S_.exp_w_;

  // Update membrane potential using exact integration scheme
  // V_m(t+h) = E_L + (V_m(t) - E_L)*exp(-h/tau_m) + (I_total/C_m)*tau_m*(1-exp(-h/tau_m))
  // where I_total = I_e + I_syn - w
  S_.V_m = P_.E_L + ( S_.V_m - P_.E_L ) * S_.exp_m_ + ( P_.I_e + I_syn - S_.w ) * S_.C1_;

  // Check for spike
  if ( S_.V_m >= P_.V_th )
  {
    // 1. Emit spike event
    this->send_spike( t );

    // 2. Set refractory period
    S_.ref_end_ = t.get_steps( ) + Time( Time::ms( P_.t_ref ) ).get_steps( );

    // 3. Reset membrane potential
    S_.V_m = P_.V_reset;

    // 4. Increment adaptation current w
    S_.w += P_.b_w;
  }
}

// Explicitly instantiate the logger
template class nest::UniversalDataLogger< iaf_bw_2001_w_k >;

} // namespace nest
