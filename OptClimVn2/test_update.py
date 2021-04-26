"""
Code to test update doing what I think it should...
"""
import tempfile
import HadCM3
import os

tdir = tempfile.tempdir
configDir = os.path.join(os.getenv("OPTCLIMTOP"), 'Configurations')
ctl = HadCM3.HadCM3(os.path.join(tdir, 'ctl'), create=True, refDirPath=os.path.join(configDir, 'xnmea'),
                    parameters={'VF1': 2.0}, name='ctlaa')
onePercent = HadCM3.HadCM3(os.path.join(tdir, 'onePer'), create=True, refDirPath=os.path.join(configDir, 'xnmeb'),
                           parameters={'VF1': 2.0}, name='1%aaa')

ctl.continueSimulation()
onePercent.continueSimulation()
##onePercent differences between ctl & 1% look much as expected so mystery why when 1% gets resubmitted it fails...
