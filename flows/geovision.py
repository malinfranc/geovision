
# This file is your entry point:
# - add you Python files and folder inside this 'flows' folder
# - add your imports
# - just don't change the name of the function 'run()' nor this filename ('geovision.py')
#   and everything is gonna be ok.
#
# Remember: everything is gonna be ok in the end: if it's not ok, it's not the end.
# Alternatively, ask for help at https://github.com/deeplime-io/onecode/issues

import onecode
from flows.global_analysis import GlobalAnalysis


def run():

    global_analysis = GlobalAnalysis(
        gdf_filename ="C:/Users/HP/Documents/geovision/Données d'entrée/Points_geochimie_AMBAZAC.geojson",
        mnt_filename = "C:/Users/HP/Documents/geovision/Données d'entrée/MNT_25M_AMBAZAC_IMAGE_CORR.tif"
    )
    global_analysis.analyze()