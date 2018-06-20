from __future__ import print_function
from spices_thickice import get_l2p_file_list, L2PSITCollection, L2pSITMap, ThickestIceMap

import os
import sys

def main(l2p_repo):
    """ Add doc here """
    
    # Get a list of thickness profiles
    l2p_file_list = get_l2p_file_list(l2p_repo)
    print("%g l2p files found" % len(l2p_file_list))

    # Create a collection of l2p sea ice thickness data
    l2p_collect = L2PSITCollection(l2p_file_list)

    # Plot the thickness values
    fig = L2pSITMap(l2p_collect)
    fig.show(block=False)

    # Plot the thickest ice location with default settings
    fig = ThickestIceMap(l2p_collect)
    fig.show(block=False)

    # Plot the thickest ice location with custom settings
    settings = dict(sit_threshold=8.0)
    fig = ThickestIceMap(l2p_collect, detection_settings=settings)
    fig.show()

if __name__ == '__main__':

    # Get the path to the sea ice thickness data repository
    # with some sanity checks
    try: 
        l2p_repo = sys.argv[1]
    except IndexError:
        sys.exit("Error: path to local l2p repository required")
    if not os.path.isdir(l2p_repo):
        raise ValueError("%s is not a valid directory" % str(l2p_repo))

    # Execute the main program 
    main(l2p_repo)