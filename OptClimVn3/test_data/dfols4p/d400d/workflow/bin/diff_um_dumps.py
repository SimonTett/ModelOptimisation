#!/usr/bin/env python
# (c) Crown copyright Met Office. All rights reserved.
"""
Compared two UM dump files by comparing instantanious data (not headers)

"""
import sys
import argparse
import um_utils.cumf
import mule

try:
    import mule
except IOError:
    sys.exit("Unable to import Mule. Ensure Scitools module is loaded")

parser = argparse.ArgumentParser(
    "Compare UM dump files", epilog=__doc__)
parser.add_argument("file1", type=str, help="First dump file.")
parser.add_argument("file2", type=str, help="Second dump file.")
args = parser.parse_args()

um_utils.cumf.COMPARISON_SETTINGS["ignore_templates"] = {'fixed_length_header': [35, 36, 37, 38, 39, 40, 41, 27, 32, 160], 'lookup': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 29]}

ff1 = mule.DumpFile.from_file(args.file1)
ff2 = mule.DumpFile.from_file(args.file2)

matching=True
failure_str=''

for ifield, field1 in enumerate(ff1.fields):

    # Only compare not time mean fields
    if field1.lbproc < 128:
    
        difference_object = um_utils.cumf.DifferenceOperator()
        difference_field = difference_object.new_field([ff1.fields[ifield], ff2.fields[ifield]])
        
        if not difference_field.data_match:
            matching=False
            failure_str = failure_str+'\nDifference detected between stash {0:d} and {1:d} and lblev {2:d} and {3:d}'.format(ff1.fields[ifield].lbuser4, ff2.fields[ifield].lbuser4, ff1.fields[ifield].lblev, ff2.fields[ifield].lblev)

if matching:
    print("[INFO]: Files %s and %s compare." % (args.file1, args.file2))
else:
    print(failure_str)
    msg = "[FAIL]: Files %s and %s do not compare." % (args.file1, args.file2)
    sys.exit(msg)
