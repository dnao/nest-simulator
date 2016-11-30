/*
 *  ac_generator.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef AC_GENERATOR_H
#define AC_GENERATOR_H

// provides AC input current

// Includes from nestkernel:
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "node.h"
#include "stimulating_device.h"

/* BeginDocumentation
   Name: ac_generator - provides AC input current
   Description:

   This device produce an ac-current which are sent by a current event.
   The parameters are
   amplitude   double -  Amplitude of sine current in pA
   offset      double -  Constant amplitude offset in pA
   phase       double -  Phase of sine current (0-360 deg)
   frequency   double -  Frequency in Hz
   4) The

   The currents are updated every time step by exact integration schemes from
   [1]

   References:
   [1] S. Rotter and M. Diesmann, Exact digital simulation of time-
   invariant linear systems with applications to neuronal modeling,
   Biol. Cybern. 81, 381-402 (1999)

   Sends: CurrentEvent

   Author: Johan Hake, Spring 2003

   SeeAlso: Device, StimulatingDevice, dc_generator
*/

namespace nest
{
class ac_generator : public Node
{

public:
  ac_generator();
  ac_generator( const ac_generator& );

  bool
  has_proxies() const
  {
    return false;
  }

  port send_test_event( Node&, rport, synindex, bool );

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

  void set_local_device_id( const index ldid );
  index get_local_device_id() const;

private:
  void init_state_( const Node& );
  void init_buffers_();
  void calibrate();

  void update( Time const&, const long, const long );


  // ------------------------------------------------------------

  struct Parameters_
  {
    double amp_;     //!< Amplitude of sine-current
    double offset_;  //!< Offset of sine-current
    double freq_;    //!< Standard frequency in Hz
    double phi_deg_; //!< Phase of sine current (0-360 deg)

    Parameters_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
    void set( const DictionaryDatum& ); //!< Set values from dicitonary
  };

  // ------------------------------------------------------------

  struct State_
  {
    double y_0_;
    double y_1_;

    State_(); //!< Sets default parameter values

    void get( DictionaryDatum& ) const; //!< Store current values in dictionary
  };

  // ------------------------------------------------------------

  struct Variables_
  {
    double omega_;   //!< Angelfrequency i rad/s
    double phi_rad_; //!< Phase of sine current (0-2Pi rad)

    // The exact integration matrix
    double A_00_;
    double A_01_;
    double A_10_;
    double A_11_;
  };

  // ------------------------------------------------------------

  StimulatingDevice< CurrentEvent > device_;
  Parameters_ P_;
  State_ S_;
  Variables_ V_;

  index local_device_id_;
};

inline port
ac_generator::send_test_event( Node& target,
  rport receptor_type,
  synindex syn_id,
  bool )
{
  device_.enforce_single_syn_type( syn_id );

  CurrentEvent e;
  e.set_sender( *this );

  return target.handles_test_event( e, receptor_type );
}

inline void
ac_generator::get_status( DictionaryDatum& d ) const
{
  P_.get( d );
  S_.get( d );
  device_.get_status( d );
}

inline void
ac_generator::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty

  // State_ is read-only

  // We now know that ptmp is consistent. We do not write it back
  // to P_ before we are also sure that the properties to be set
  // in the parent class are internally consistent.
  device_.set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
}

inline void
ac_generator::set_local_device_id( const index ldid )
{
  local_device_id_ = ldid;
}

inline index
ac_generator::get_local_device_id() const
{
  return local_device_id_;
}
}

#endif // AC_GENERATOR_H
