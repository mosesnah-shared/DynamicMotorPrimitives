"""
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

class CanonicalSystem:
    """
        Implementation of the canonical dynamical system as described in Dr. Stefan Schaal's (2002) paper
    """

    def __init__( self, dt, ax = 1.0, pattern = "discrete" ):
        """
            Default values from Schaal (2012)

            [INPUT]
                [VAR NAME]    [TYPE]     [DESCRIPTION]
                (1) dt        float      The timestep
                (2) ax        float      A gain term on the dynamical system
                (3) pattern   string     Either 'discrete' or 'rhythmic'
        """
        self.dt      = dt
        self.ax      = ax
        self.pattern = pattern

        if   pattern == "discrete":
            self.step     = self.step_discrete
            self.run_time = 1.0
        elif pattern == "rhythmic":
            self.step     = self.step_rhythmic
            self.run_time = 2 * np.pi
        else:
            raise Exception( "Invalid pattern type specified: Please specify rhythmic or discrete." )

        self.time_steps = int( self.run_time / self.dt )
        self.reset_state()

    def rollout( self, **kwargs ):
        """
            Generate x for open loop movements.
        """
        time_steps = int( self.time_steps / kwargs[ "tau" ] ) if "tau" in kwargs \
                     else self.time_steps

        self.x_track = np.zeros( time_steps )

        self.reset_state()
        for t in range( time_steps ):
            self.x_track[ t ] = self.x
            self.step( **kwargs )

        return self.x_track

    def reset_state( self ):
        """
            Reset the system state
        """
        self.x = 1.0

    def step_discrete( self, tau = 1.0, error_coupling = 1.0):
        """
            Generate a single step of x for discrete (potentially closed) loop movements.
            Decaying from 1 to 0 according to dx = -ax * x.

            [INPUT]
                [VAR NAME]          [TYPE]     [DESCRIPTION]
                (1) tau             float      gain on execution time increase tau to make the system execute faster
                (2) error_coupling   float      slow down if the error is > 1

        """
        self.x += ( -self.ax * self.x * error_coupling ) * tau * self.dt
        return self.x

    def step_rhythmic( self, tau = 1.0, error_coupling = 1.0):
        """
            Generate a single step of x for rhythmic closed loop movements.
            Decaying from 1 to 0 according to dx = -ax*x.

            [INPUT]
                [VAR NAME]          [TYPE]     [DESCRIPTION]
                (1) tau             float      gain on execution time increase tau to make the system execute faster
                (2) error_coupling   float      slow down if the error is > 1
        """
        self.x += ( 1 * error_coupling * tau ) * self.dt
        return self.x



if __name__ == "__main__":

    # mode    = "discrete"
    # [BACKUP] [MOSES]
    mode    = "rhythmic"
    dt      = 0.001


    cs      = CanonicalSystem( dt, pattern = mode )
    x_track1 = cs.rollout()
    cs.reset_state()

    fig, ax1 = plt.subplots(figsize=(6, 3))

    if   mode == "discrete":

        # test error coupling
        t_total    = 1
        time_steps = int( t_total/dt )
        x_track2   = np.zeros( time_steps )
        err        = np.zeros( time_steps )

        err[ 200 : 400 ] = 2
        err_coup = 1.0 / ( 1 + err )
        for i in range( time_steps ):
            x_track2[ i ] = cs.step( error_coupling = err_coup[ i ] )



        ax1.plot( x_track1, lw = 2 )
        ax1.plot( x_track2, lw = 2 )

        plt.legend( [ "Normal Rollout", "Error Coupling" ] )
        ax2 = ax1.twinx( )
        ax2.plot( err, "r-", lw = 2)
        plt.legend( [ "error" ], loc = "lower right")
        plt.ylim( 0, 3.5)


        for t1 in ax2.get_yticklabels():
            t1.set_color("r")

        plt.tight_layout()

    elif mode == "rhythmic":
        ax1.plot( x_track1, lw = 2 )
        plt.legend( [ "Normal Rollout" ], loc = "lower right")


    plt.grid()
    plt.xlabel( "Time (s)" )
    plt.ylabel( "x" )
    plt.title( "Canonical system - {0:s}".format( mode ) )
    plt.show()
