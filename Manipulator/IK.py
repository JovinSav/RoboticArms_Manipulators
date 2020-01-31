# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:43:22 2020

@author: nivoj
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import Manipulator as Manup


Arm=Manup.Arm()
Arm.SetArmDL()
J,X=Arm.Workspace()
Arm.FK 

y=Arm.Training(15000)


#plt.scatter(J[:,0],J[:,1])
#plt.scatter(y[:,0],y[:,1])
#
#
#
X_=[-0.340678,0.202321,0]
y_new=Arm.IK(X_)
