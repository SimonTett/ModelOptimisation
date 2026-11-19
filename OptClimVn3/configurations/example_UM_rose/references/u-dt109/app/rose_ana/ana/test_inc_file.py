#!/usr/bin/env python
# *****************************COPYRIGHT*******************************
# (C) Crown copyright Met Office. All rights reserved.
# For further details please refer to the file COPYRIGHT.txt
# which you should have received as part of this distribution.
# *****************************COPYRIGHT*******************************

from rose.apps.rose_ana import AnalysisTask

class TestIncFile(AnalysisTask):
    """Analysis task to test increment budgets"""

    def run_analysis(self):
        infile = self.options['file']
        msg = '{} contains the following output:'.format(infile)
        self.parent.reporter(msg, prefix='[INFO] ')
        passed = True
        with open(infile, 'r') as ifile:
            for line in ifile:
                self.parent.reporter(line, prefix='')
                if '[FAIL]' in line:
                    passed = False
        self.passed = passed
