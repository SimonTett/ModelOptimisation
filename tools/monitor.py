#!/bin/env python
"""
 Monitor a run
"""
import argparse
import StudyConfig
import matplotlib.pyplot as plt 
import os

parser=argparse.ArgumentParser(description="Plot Current Study")
parser.add_argument("jsonFile",help="json file that defines the study")
args=parser.parse_args()
jsonFile=os.path.expanduser(os.path.expandvars(args.jsonFile))
config= StudyConfig.readConfig(filename=jsonFile,ordered=True) # parse the jsonFile.
fig=plt.figure("Monitor",figsize=[11.7,8.3])
fig.clear()
ax=fig.add_subplot(121)
config.cost().plot(ax=ax,marker='s')
ax.set_title("Cost")
ax.margins(0.1,0.1)

ax=fig.add_subplot(122)
config.parameters(normalise=True).plot(ax=ax,marker='s')
ax.set_title("Normalised Parameter")
ax.margins(0.1,0.1)
outfile=os.path.splitext(jsonFile)[0]+"_mon.png"

fig.tight_layout()
fig.savefig(outfile) # should modify to write to directory rather than where running.
fig.show()

