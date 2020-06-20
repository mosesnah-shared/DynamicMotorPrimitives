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


from .CanonicalSystem import CanonicalSystem


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
        if isinstance( y0, (int, float) ):
            y0 = np.ones( self.nDMPs ) * y0

        self.y0 = y0

        if isinstance( goal, (int, float) ):
            goal = np.ones( self.nDMPs ) * goal

        self.goal = goal

        if w is None:
            # default is f = 0
            w = np.zeros( ( self.nDMPs, self.nBFs ) )

        self.w  = w
        self.ay = np.ones( nDMPs ) * 25.0 if ay is None else ay                 # Schaal 2012
        self.by = self.ay / 4.0           if by is None else by                 # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem( dt = self.dt, **kwargs )
        self.timesteps = int( self.cs.runTime / self.dt )

        # set up the DMP system
        self.resetState()

    def checkOffset( self ):
        """
            Check to see if initial position and goal are the same
            if they are, offset slightly so that the forcing term is not 0
        """

        for d in range( self.nDMPs ):
            if abs( self.y0[ d ] - self.goal[ d ] ) < 1e-4:
                self.goal[ d ] += 1e-4

    def genFrontTerm( self, x, dmpNum ):
        raise NotImplementedError( )

    def genGoal( self, yDesired ):
        raise NotImplementedError( )

    def genPsi(self):
        raise NotImplementedError( )

    def genWeights( self, fTarget ):
        raise NotImplementedError( )

    def imitatePath(self, yDesired, isPlot = False):
        """
            Takes in a desired trajectory and generates the set of system parameters that best realize this path.

            [INPUT]
                [VAR NAME]    [TYPE]        [DESCRIPTION]
                yDesired      list/array    The desired trajectories of each DMP should be shaped [nDMPs, runTime]
        """

        # set initial state and goal
        if yDesired.ndim == 1:
            yDesired = yDesired.reshape( 1, len( yDesired ) )
        self.y0       = yDesired[ :, 0 ].copy( )
        self.yDesired = yDesired.copy()
        self.goal     = self.genGoal( yDesired )

        # self.checkOffset()


        path = np.zeros( ( self.nDMPs, self.timesteps ) )
        x    = np.linspace( 0, self.cs.runTime, yDesired.shape[ 1 ] )

        for d in range( self.nDMPs ):
            path_gen = scipy.interpolate.interp1d( x, yDesired[ d ] )
            for t in range( self.timesteps ):
                path[ d, t ] = path_gen(t * self.dt)

        yDesired   = path
        dyDesired  = np.gradient(  yDesired, axis = 1) / self.dt                # Calculate velocity of yDesired with central differences
        ddyDesired = np.gradient( dyDesired, axis = 1) / self.dt                # Calculate acceleration of yDesired with central differences

        fTarget    = np.zeros( ( yDesired.shape[ 1 ], self.nDMPs ) )

        # find the force required to move along this trajectory
        for d in range(self.nDMPs):
            fTarget[:, d] = ddyDesired[ d ] - self.ay[ d ] * (
                self.by[d] * (self.goal[ d ] - yDesired[ d ]) - dyDesired[ d ]
            )

        # efficiently generate weights to realize fTarget
        self.genWeights( fTarget )

        if isPlot:
            # plot the basis function activations

            plt.figure( )
            plt.subplot( 211 )
            psiTrack = self.genPsi( self.cs.rollout( ) )
            plt.plot( psiTrack )
            plt.title( "Basis Functions" )

            # plot the desired forcing function vs approx
            for i in range( self.nDMPs ):
                plt.subplot( 2, self.nDMPs, self.nDMPs + 1 + i )
                plt.plot( fTarget[ :, i ], "--", label = "fTarget %i" % i)

            for i in range( self.nDMPs ):
                plt.subplot( 2, self.nDMPs, self.nDMPs + 1 + i )
                print( "w shape: ", self.w.shape )
                plt.plot(
                    np.sum( psiTrack * self.w[ i ], axis = 1) * self.dt,
                    label = "w*psi %i" % i )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.resetState()
        return yDesired

    def rollout( self, timesteps = None, **kwargs ):
        """
            Generate a system trial, no feedback is incorporated.
        """

        self.resetState()

        if timesteps is None:
            timesteps = int( self.timesteps / kwargs[ "tau" ] ) if "tau" in kwargs else self.timesteps

        # set up tracking vectors
        yTrack   = np.zeros( ( timesteps, self.nDMPs ) )
        dyTrack  = np.zeros( ( timesteps, self.nDMPs ) )
        ddyTrack = np.zeros( ( timesteps, self.nDMPs ) )

        for t in range(timesteps):

            yTrack[t], dyTrack[t], ddyTrack[t] = self.step( **kwargs )            # run and record timestep

        return yTrack, dyTrack, ddyTrack

    def resetState( self ):
        """
            Reset the system state
        """
        self.y   = self.y0.copy()
        self.dy  = np.zeros( self.nDMPs )
        self.ddy = np.zeros( self.nDMPs )
        self.cs.resetState()

    def step( self, tau = 1.0, error = 0.0, externalForce = None):
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
        psi = self.genPsi( x )

        for d in range( self.nDMPs ):

            # generate the forcing term
            f = self.genFrontTerm( x, d ) * ( np.dot( psi, self.w[ d ] ) ) / np.sum( psi )

            # DMP acceleration
            self.ddy[ d ] = (
                self.ay[ d ] * (self.by[ d ] * ( self.goal[ d ] - self.y[ d ] ) - self.dy[ d ] ) + f
            )

            if externalForce is not None:
                self.ddy[ d ] += externalForce[ d ]

            self.dy[ d ] += self.ddy[ d ] * tau * self.dt * errorCoupling
            self.y[ d ]  +=  self.dy[ d ] * tau * self.dt * errorCoupling

        return self.y, self.dy, self.ddy
