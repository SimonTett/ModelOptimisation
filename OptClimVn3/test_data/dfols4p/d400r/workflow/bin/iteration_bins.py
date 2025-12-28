#!/usr/bin/python

# Program to put the number of iterations of the solver into bins

import sys, os

# ****** MAIN PROGRAM ******

# Set up the input arguments
try:
    this_exec = sys.argv[0]
    input_file = sys.argv[1]
    runid = sys.argv[2]
except IndexError:
    print "Program to put the number of iterations of the solver into bins."
    print
    print 'Syntax:  %s input_file runid' % (this_exec)
    print
    sys.exit(1)
    
# Check to see if the iterations file exists
iter_file=os.path.join(os.environ['ROSE_DATA'], runid+'_iterations.list')
bins=[]
if os.path.exists(iter_file):
    
    # Load in the bins
    f = open(iter_file, 'r')
    for line in f:
        bins.append(int(line))
    f.close()

else:
    
    # Make a blank list
    for a in range(200):
        bins.append(0)

# Open the file
f = open(input_file, 'r')

# Loop over all the lines
for line in f:
    thiscount=bins[int(line)]
    thiscount=thiscount + 1
    bins[int(line)] = thiscount

# Close the file
f.close()

# Output the file
f = open(iter_file,'w')
for bin in bins:
   f.write(str(bin)+'\n')
f.close
