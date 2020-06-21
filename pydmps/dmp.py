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

import numpy             as np
import matplotlib.pyplot as plt
import scipy.interpolate


from cs import CanonicalSystem


class DMPs( object ):
    """
        Implementation of Dynamic Motor Primitives, as described in Dr. Stefan Schaal's (2002) paper.
    """

    def __init__( self, nDMPs, nBFs, dt = 0.01, y0 = 0, goal = 1, w = None, ay = None, by = None, **kwargs ):
        """
            Default values from Schaal (2012)

            [INPUT]
                [VAR NAME]    [TYPE]     [DESCRIPTION]
                (1) nDMPs     int      Number of dynamic motor primitives
                (2) nBFs      int      Number of basis functions per DMP
                (3) dt         float    Timestep for simulation
                (4) y0         list     Initial state of DMPs
                (5) goal       list     Goal state of DMPs
                (6) w          list     Tunable parameters, control amplitude of basis functions
                (7) ay         int      Gain on attractor term y dynamics
                (8) by         int      Gain on attractor term y dynamics
        """


        self.nDMPs = nDMPs
        self.nBFs  = nBFs
        self.dt    = dt
        self.y0    = y0   * np.ones( self.nDMPs ) if isinstance(   y0, ( int, float ) ) else y0
        self.goal  = goal * np.ones( self.nDMPs ) if isinstance( goal, ( int, float ) ) else goal
        self.w     = np.zeros( ( self.nDMPs, self.nBFs ) ) if  w is None else w
        self.ay    = np.ones( nDMPs ) * 25.0               if ay is None else ay   # Schaal 2012
        self.by    = self.ay / 4.0                         if by is None else by   # Schaal 2012, 0.25 makes it a critically damped system

        # set up the CS
        self.cs = CanonicalSystem( dt = self.dt, **kwargs )
        self.timesteps = int( self.cs.runTime / self.dt )

        # set up the DMP system
        self.reset_state()

    def check_offset( self ):
        """
            Check to see if initial position and goal are the same
            if they are, offset slightly so that the forcing term is not 0
        """

        for d in range( self.nDMPs ):
            if abs( self.y0[ d ] - self.goal[ d ] ) < 1e-4:
                self.goal[ d ] += 1e-4

    def gen_front_term( self, x, dmpNum ):
        raise NotImplementedError( )

    def gen_goal( self, y_des ):
        raise NotImplementedError( )

    def gen_psi(self):
        raise NotImplementedError( )

    def gen_weights( self, f_target ):
        raise NotImplementedError( )

    def imitatePath(self, y_des, isPlot = False):
        """
            Takes in a desired trajectory and generates the set of system parameters that best realize this path.

            [INPUT]
                [VAR NAME]        [TYPE]        [DESCRIPTION]
                (1) y_des      list/array    The desired trajectories of each DMP should be shaped [nDMPs, runTime]
        """

        # set initial state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape( 1, len( y_des ) )
        self.y0       = y_des[ :, 0 ].copy( )
        self.y_des = y_des.copy( )
        self.goal     = self.gen_goal( y_des )

        # self.check_offset()


        path = np.zeros( ( self.nDMPs, self.timesteps ) )
        x    = np.linspace( 0, self.cs.runTime, y_des.shape[ 1 ] )

        for d in range( self.nDMPs ):
            path_gen = scipy.interpolate.interp1d( x, y_des[ d ] )
            for t in range( self.timesteps ):
                path[ d, t ] = path_gen(t * self.dt)

        y_des   = path
        dy_des  = np.gradient(  y_des, axis = 1) / self.dt                # Calculate velocity of y_des with central differences
        ddy_des = np.gradient( dy_des, axis = 1) / self.dt                # Calculate acceleration of y_des with central differences

        f_target    = np.zeros( ( y_des.shape[ 1 ], self.nDMPs ) )

        # find the force required to move along this trajectory
        for d in range(self.nDMPs):
            f_target[:, d] = ddy_des[ d ] - self.ay[ d ] * (
                self.by[d] * (self.goal[ d ] - y_des[ d ]) - dy_des[ d ]
            )

        # efficiently generate weights to realize f_target
        self.gen_weights( f_target )

        if isPlot:
            # plot the basis function activations

            plt.figure( )
            plt.subplot( 211 )
            psi_track = self.gen_psi( self.cs.rollout( ) )
            plt.plot( psi_track )
            plt.title( "Basis Functions" )

            # plot the desired forcing function vs approx
            for i in range( self.nDMPs ):
                plt.subplot( 2, self.nDMPs, self.nDMPs + 1 + i )
                plt.plot( f_target[ :, i ], "--", label = "f_target %i" % i)

            for i in range( self.nDMPs ):
                plt.subplot( 2, self.nDMPs, self.nDMPs + 1 + i )
                print( "w shape: ", self.w.shape )
                plt.plot(
                    np.sum( psi_track * self.w[ i ], axis = 1) * self.dt,
                    label = "w*psi %i" % i )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.reset_state()
        return y_des

    def rollout( self, timesteps = None, **kwargs ):
        """
            Generate a system trial, no feedback is incorporated.
        """

        self.reset_state()

        if timesteps is None:
            timesteps = int( self.timesteps / kwargs[ "tau" ] ) if "tau" in kwargs else self.timesteps

        # set up tracking vectors
        yTrack   = np.zeros( ( timesteps, self.nDMPs ) )
        dyTrack  = np.zeros( ( timesteps, self.nDMPs ) )
        ddyTrack = np.zeros( ( timesteps, self.nDMPs ) )

        for t in range(timesteps):

            yTrack[t], dyTrack[t], ddyTrack[t] = self.step( **kwargs )            # run and record timestep

        return yTrack, dyTrack, ddyTrack

    def reset_state( self ):
        """
            Reset the system state
        """
        self.y   = self.y0.copy()
        self.dy  = np.zeros( self.nDMPs )
        self.ddy = np.zeros( self.nDMPs )
        self.cs.reset_state()

    def step( self, tau = 1.0, error = 0.0, externalForce = None ):
        """
            Run the DMP system for a single timestep.

            [INPUT]
                [VAR NAME]    [TYPE]        [DESCRIPTION]
                tau            float    Scales the timestep increase tau to make the system execute faster
                error          float    Optional system feedback
        """

        errorCoupling = 1.0 / ( 1.0 + error )
        # run canonical system
        x = self.cs.step( tau = tau, errorCoupling = errorCoupling )

        # generate basis function activation
        psi = self.gen_psi( x )

        for d in range( self.nDMPs ):

            # generate the forcing term
            f = self.gen_front_term( x, d ) * ( np.dot( psi, self.w[ d ] ) ) / np.sum( psi )

            # DMP acceleration
            # DMP equation is as following:
            # tau * y'' + ay * y' + ayby * (y-g) = f
            self.ddy[ d ] = (
                self.ay[ d ] * (self.by[ d ] * ( self.goal[ d ] - self.y[ d ] ) - self.dy[ d ] ) + f
            )

            if externalForce is not None:
                self.ddy[ d ] += externalForce[ d ]

            self.dy[ d ] += self.ddy[ d ] * tau * self.dt * errorCoupling
            self.y[ d ]  +=  self.dy[ d ] * tau * self.dt * errorCoupling

        return self.y, self.dy, self.ddy
