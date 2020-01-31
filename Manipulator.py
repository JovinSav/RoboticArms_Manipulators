# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:41:44 2020

@author: nivoj
"""
#################################################
from sympy import symbols
from sympy import *
from sympy.matrices import eye
from sympy.matrices import *
#################################################
import numpy as np
import math
from sympy.utilities.lambdify import implemented_function
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
init_printing(use_unicode=True)
################################################
'''Torch Module import'''
################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
###############################################
from sklearn.preprocessing import StandardScaler





class Manipulator():
    def __init__(self):
        self.no_joint=int(input("interger 0--N?\t"))
        j=0
        while(1):
            self.config=str(input("config string of 'RRR..PPP...'upto N num ?\t")).upper()
            self.__Joint=[]
            if len(self.config)==self.no_joint:
               r=0
               p=0
               for i in self.config:
                   if ((i=='R') or (i=='P')):
                       j+=1
                       
                       if i=='R':
                           r+=1
                           i=f"ROTATIONAL JOINT:{r}"
                       elif i=='P':
                           p+=1
                           i=f"PRISMATIC JOINT:{p}"
                       self.__Joint.append(i)
                       print(self.__Joint,j)
                   else:
                     j=0
                     self.__Joint=[]
                     print(f"Worng Confingratton {self.config}")
                     break
               if j>0:
                   print(f"Manipulator has Joint:={self.no_joint}  corret Confingratton:= {self.config}")
                   self.Joint_Variables()
                   print(self.__joint_variable)
                   #private variable
                   self.__DH=self.__Set_all()
                   self.DH=self.__DH
                   print(*self.__DH)
                   self.jv=self.__joint_variable
                   self.Tb=self.__Tb()
                   self.Ttheta=self.__Ttheta()
                   self.Ta=self.__Ta()
                   self.Talpha=self.__Talpha()
                   self.Tspace=self.__Ti()
                   self.joint=self.__Joint
                   break
            else:
                print(f"Input corret Confingratton {self.config}")
    def __Set_all(self):
        DH=[]
        for i in self.__Joint:
            b=i.split(':')
            if b[0]=="ROTATIONAL JOINT":
                linka=float(input(f"Link Length a{b[1]} of {i} in meters:=  \t"))
                linkd=float(input(f"Link Length d{b[1]} of {i} in meters:=  \t"))
                linkth=self.__joint_variable[self.__Joint.index(i)]
                linkalpha=float(input(f"Link alpha of {b} :=  \t"))
            elif b[0]=="PRISMATIC JOINT":
                linka=float(input(f"Link Length a{b[1]} of {i} in meters:=  \t"))
                linkd=self.__joint_variable[self.__Joint.index(i)]
                linkth=float(input(f"Link theta of {i} :=  \t"))
                linkalpha=float(input(f"Link alpha of {i} :=  \t"))
            DH.append([linka,linkalpha,linkd,linkth])
        return DH
            
           
    def __Set_Joint_Len(self):#private function setting link lengths
        linklen=[]
        p=1
        for i in self.__Joint:
            b=i.split(':')
            if b[0]=="ROTATIONAL JOINT":
                link=float(input(f"Link Length of {i} in meters:=  \t"))
                linklen.append(link)
                p+=1
            elif b[0]=="PRISMATIC JOINT":
                link=0.0
                linklen.append(link)
                
        
        return linklen
    def Joint_Variables(self):
        joint=[]
        t=0
        p=0
        for i in self.__Joint:
            i=i.split(':')
            if i[0]=="ROTATIONAL JOINT":
                t+=1
                joint.append(symbols(f"theta{t}"))
            elif i[0]=="PRISMATIC JOINT":
                p+=1
                joint.append(symbols(f"b{p}"))
            elif i[0]=="SPACIAL JOINT":
                p+=1
                joint.append(symbols(f"s{p}"))
        self.__joint_variable=joint
        joint=[]
        t=0
        p=0
        return self.__joint_variable
    def __Tb(self):
         jv=self.__joint_variable
         DH=self.__DH
         a=self.__Joint
         dummy=[]
         for i in self.__Joint:
             b=i
             i=i.split(':')
             if i[0]=="PRISMATIC JOINT":
                 tb=eye(4)
                 tb[2,3]=jv[a.index(b)]
                 
                 dummy.append(tb)
             elif i[0]=="ROTATIONAL JOINT":
                 k=DH[a.index(b)][2]
                 tb=eye(4)
                 tb[2,3]=k
                 dummy.append(tb)
         return dummy
    def __Ttheta(self):
         jv=self.__joint_variable
         DH=self.__DH
         a=self.__Joint
         dummy=[]
         for i in self.__Joint:
             b=i
             i=i.split(':')
             print(b)
             if i[0]=="ROTATIONAL JOINT":
                 th=eye(4)
                 th[0,0]=cos(jv[a.index(b)])
                 th[0,1]=-1*sin(jv[a.index(b)])
                 
                 th[1,0]=sin(jv[a.index(b)])
                 th[1,1]=cos(jv[a.index(b)])
                
                 dummy.append(th)
             elif i[0]=="PRISMATIC JOINT":
                 th=eye(4)
                 th[0,0]=math.cos(DH[a.index(b)][3])
                 th[0,1]=-1*math.sin(DH[a.index(b)][3])
                 th[1,0]=math.sin(DH[a.index(b)][3])
                 th[1,1]=math.cos(DH[a.index(b)][3])
                 dummy.append(th)
                 
         return dummy
    def __Ta(self):
         link=self.__DH
         dummy=[]
         for i in link:
             ta=eye(4)
             ta[0,3]=link[0][0]
             dummy.append(ta)
         return dummy
    def __Talpha(self):
        dummy=[]
        self.__linkalpha=self.__DH
        for i in self.__linkalpha:
            ta=eye(4)
            ta[1,1]=cos(math.radians(i[1]))
            ta[1,2]=-1*sin(math.radians(i[1]))
            ta[2,1]=sin(math.radians(i[1]))
            ta[2,2]=cos(math.radians(i[1]))
            dummy.append(ta)
        return dummy
    def __Ti(self):
        tb=self.Tb
        tth=self.Ttheta
        ta=self.Ta
        tal=self.Talpha
        dummy=[]
        for i in range(self.no_joint):
            T=tth[i]*tb[i]*ta[i]*tal[i]
            dummy.append(T)
        return dummy
    def ForwardMatrix(self):
        T=self.Tspace
        t=eye(4)
        
        for i in range(len(T)):
            t*=T[i]
        return t
    def Reset(self):
        self.Joint_Variables()
        print(self.__joint_variable)
        self.jv=self.__joint_variable
        self.Tb=self.__Tb()
        self.Ttheta=self.__Ttheta()
        self.Ta=self.__Ta()
        self.Talpha=self.__Talpha()
        self.Tspace=self.__Ti()
        self.joint=self.__Joint
        
            
        
             
class Arm(Manipulator):
    def __init__(self):
        start_time = time.time()
        super(Arm,self).__init__()
        print(f"MANIPULATOR created in {time.time()-start_time} seconds\t",'\n'*5)
        u=np.array(list(self.__Set_ranges()))
        self.__rangeSpace=u.T
        self.__points=np.array(list(self.__Gen_Joint_Val()))
        
    def __Set_ranges(self):#private funtion to Set Ranges 
         #z=FK[11]
         st = time.time()
         self.__ranges=[]
         a=self.joint
         for i in a:
             b=i.split(':')
             if b[0]=="ROTATIONAL JOINT":
                 a=list(map(float,input(f"min and max of JV {self.jv[self.joint.index(i)]} in degrees?\t").split(' ')))
                 a=list(map(math.radians,a))
             elif b[0]=="PRISMATIC JOINT":
                 a=list(map(float,input(f"min and max of JV {self.jv[self.joint.index(i)]} in meters?\t").split(' ')))
             self.__ranges.append(a)
         self.ranges=self.__ranges
         a=[]
         for i in self.ranges:
             u=np.random.uniform(i[0],i[1],(1000,))
             a=u.ravel()
             yield a
         et= time.time()
         print(f"Ranges set CREATED IN {et-st} seconds")
         
         
    def __Gen_Joint_Val(self):
         st = time.time()
         FK=self.ForwardMatrix()
         self.get_sys=list(FK.free_symbols)
         x=FK[3]
         y=FK[7]
         z=FK[11]
         self.x=x
         self.y=y
         self.z=z
         self.FK=FK
         for i in self.__rangeSpace:
             a={}
             for p in range(len(i)):
                 a[self.jv[p]]=float(i[p])
             
             j=float(self.x.subs(a))
             u=float(self.y.subs(a))
             r=float(self.z.subs(a))
             p=[j,u,r]
             yield p
         
         et= time.time()
         print(f"points CREATED IN {et-st} seconds")
         
    def Workspace(self):
       return self.__rangeSpace,self.__points
    def JspaceRadtoDegree(self,x=True,Y=0):
        if x==True:
            o=self.__rangeSpace
        else:
            o=Y
        p=[]
        
        for i in o:
            r=[]
            for j in range(len(i)):
                q,w=self.joint[j].split(':')
                if q=="ROTATIONAL JOINT":
                    k=math.degrees(i[j])
                elif q=="PRISMATIC JOINT":
                    k=i[j]
                r.append(k)
            p.append(r)
        o=[]
        r=[]
        return np.array(p)
    def JspaceToCoord(self,X):
        corrd=[]
        po=[]
        for i in X:
            a={}
            x=self.x
            y=self.y
            z=self.z
            for p in range(len(i)):
                a[self.jv[p]]=float(i[p])
            po.append(a)
            j=float(x.subs(a))
            u=float(y.subs(a))
            r=float(z.subs(a))
            corrd.append([j,u,r])
        return np.array(corrd)
    def SetArmDL(self,layers=[20,80,80,80,20]):
        st = time.time()
        self.__scx=StandardScaler()
        self.__X=self.__scx.fit_transform(self.__points)
        self.__scy=StandardScaler()
        self.__Y=self.__scy.fit_transform(self.__rangeSpace)
        Dout=len(self.__rangeSpace[0,:])
        Din=len(self.__points[0,:])
        self.__Model=ArmNN(Din,Dout,layers)
        self.__criteria=nn.MSELoss()   
        et= time.time()
        print(f"ArmDLModel CREATED \n{self.__Model.parameters} \n Time Taken {et-st} seconds  ")
    def Training(self,epochs=5000,lr=0.01):
        optimiser=torch.optim.Adam(self.__Model.parameters(),lr=lr)
        X=torch.FloatTensor(self.__X)
        Y=torch.FloatTensor(self.__Y)
        print(X.shape,Y.shape)
        epoch=epochs
        losses=[]
        st=time.time()
        for i in range(epoch+1):
            y_pred=self.__Model.forward(X)
            losse=self.__criteria(y_pred,Y)
            losses.append(losse)
            print(f"time:{time.time()-st} secs epoch: {i} loss: {losse.item()}")
            optimiser.zero_grad()
            losse.backward()
            optimiser.step()
        plt.plot(range(epoch+1),losses)
        losses=[]
        y_=y_pred.detach().numpy()
        y_=self.__scy.inverse_transform(y_)
        return y_
    def IK(self,X):
        X=np.array(X).reshape(-1,3)
        X1=self.__scx.transform(X)
        X1=torch.FloatTensor(X1)
        y_pred=self.__Model.forward(X1)
        y_=y_pred.detach().numpy()
        y_=self.__scy.inverse_transform(y_)
        print(f"All Values in Radians")
        return y_
class ArmNN(nn.Module):
     def __init__(self,Din,Dout,layers):
         super(ArmNN,self).__init__()
         self.layerlist=[]
         for i in layers:
             self.layerlist.append(nn.Linear(Din,i))
             self.layerlist.append(nn.Tanh())# activation function
             Din=i
         self.layerlist.append(nn.Linear(layers[-1],Dout))
         self.net=nn.Sequential(*self.layerlist)
     
     def forward(self,x):
         x=self.net(x)
         return x            
        
            
            
         
         
             
         

#a=Arm()
#ranges=a.Get_ranges() 
#points=a.Workspace()
#plt.scatter(points[:,0],points[:,1])        
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(points[:,0].reshape(-1,1),points[:,1].reshape(-1,1),points[:,2].reshape(-1,1))
#plt.show()        