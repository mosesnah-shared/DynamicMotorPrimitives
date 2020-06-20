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

# from pydmps.DMP import DMPs
from .DMP import DMPs
import numpy             as np
import matplotlib.pyplot as plt


class DMP_Rhythmic( DMPs ):
    """
        An implementation of discrete DMPs
    """

    def __init__( self, **kwargs ):
        """
        """

        # call super class constructor
        super( DMP_Rhythmic, self).__init__( pattern = "rhythmic", **kwargs )
        self.genCenters()

        # set variance of Gaussian basis functions, trial and error to find this spacing
        self.h = np.ones( self.nBFs ) * self.nBFs  # 1.75

        self.checkOffset()

    def genCenters(self):
        """
            Set the centre of the Gaussian basis functions be spaced evenly throughout run time
        """

        c = np.linspace( 0, 2 * np.pi, self.nBFs + 1 )
        c = c[ 0:-1 ]
        self.c = c

    def genFrontTerm( self, x, dmpNum ):
        """
            Generates the front term on the forcing term.
            For rhythmic DMPs it's non-diminishing, so this function is just a placeholder to return 1.

            [INPUT]
                [VAR NAME]    [TYPE]     [DESCRIPTION]
                (1) x         float      The current value of the canonical system
                (2) dmpNum    int        The index of the current dmp

        """

        if isinstance( x, np.ndarray ):
            return np.ones( x.shape )

        return 1

    def genGoal( self, yDesired ):
        """
            Generate the goal for path imitation.
            For rhythmic DMPs the goal is the average of the desired trajectory.

            [INPUT]
                [VAR NAME]    [TYPE]     [DESCRIPTION]
                (1) yDesired      np.array   The desired trajectory to follow
        """

        goal = np.zeros( self.nDMPs )
        for n in range( self.nDMPs ):
            numIdx = ~np.isnan( yDesired[ n ] )  # ignore nan's when calculating goal
            goal[ n ] = 0.5 * (yDesired[ n, numIdx ].min() + yDesired[ n, numIdx ].max( ) )

        return goal

    def genPsi( self, x ):
        """
            Generates the activity of the basis functions for a given canonical system state or path.

            [INPUT]
                [VAR NAME]    [TYPE]     [DESCRIPTION]
                (1) x      float, array   The canonical system state or path

        """

        if isinstance( x, np.ndarray ):
            x = x[ :, None ]
        return np.exp( self.h * ( np.cos( x - self.c ) - 1 ) )

    def genWeights( self, fTarget ):
        """
            Generate a set of weights over the basis functions such that the target forcing term trajectory is matched.

            [INPUT]
                [VAR NAME]    [TYPE]     [DESCRIPTION]
                (1) fTarget   np.array   The desired forcing term trajectory
        """

        # calculate x and psi
        xTrack   = self.cs.rollout( )
        psiTrack = self.genPsi( xTrack )

        # efficiently calculate BF weights using weighted linear regression
        for d in range( self.nDMPs ):
            for b in range( self.nBFs ):
                self.w[ d, b ] = np.dot( psiTrack[ :, b ], fTarget[ :, d ]) / (
                    np.sum( psiTrack[ :, b ] ) + 1e-10 )



if __name__ == "__main__":

    # test normal run
    dmp = DMP_Rhythmic( nDMPs = 1, nBFs = 10, w = np.zeros( (1, 10 ) ) )

    yTrack, dyTrack, ddyTrack = dmp.rollout()

    plt.figure(1, figsize=(6, 3))
    plt.plot(np.ones(len(yTrack)) * dmp.goal, "r--", lw=2)
    plt.plot(yTrack, lw=2)
    plt.title("DMP system - no forcing term")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["goal", "system state"], loc="lower right")
    plt.tight_layout()

    # test imitation of path run
    plt.figure( 2, figsize=( 6, 4 ) )
    nBFs = [10, 30, 50, 100, 10000]


    path1 = np.sin(np.arange(0, 2 * np.pi, 0.01) * 5)                           # a straight line to target
    path2 = np.zeros( path1.shape )                                               # a strange path to target
    path2[int(len(path2) / 2.0) :] = 0.5

    for ii, bfs in enumerate(nBFs):
        dmp = DMP_Rhythmic(nDMPs=2, nBFs=bfs)

        dmp.imitatePath(yDesired=np.array([path1, path2]))
        yTrack, dyTrack, ddyTrack = dmp.rollout()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(yTrack[:, 0], lw=2)
        plt.subplot(212)
        plt.plot(yTrack[:, 1], lw=2)

    plt.subplot(211)
    a = plt.plot(path1, "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend([a[0]], ["desired path"], loc="lower right")
    plt.subplot(212)
    b = plt.plot(path2, "r--", lw=2)
    plt.title("DMP imitate path")
    plt.xlabel("time (ms)")
    plt.ylabel("system trajectory")
    plt.legend(["%i BFs" % i for i in nBFs], loc="lower right")

    plt.tight_layout()
    plt.show()
