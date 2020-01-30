# RoboticArms_Manipulators
*This Repository is used for beginner to do IK  of any Manipulator Robots of Configurations of with combinations of Rotational and Prismatics Joints*

# Problem Statement


## About Manipulators
In robotics, a manipulator is a device used to manipulate materials without direct physical contact by the operator.The applications were originally for dealing with radioactive or biohazardous materials, using robotic arms, or they were used in inaccessible places. In more recent developments they have been used in diverse range of applications including welding automation,robotic surgery and in space. It is an arm-like mechanism that consists of a series of segments, usually sliding or jointed called cross-slides,which grasp and move objects with a number of degrees of freedom .*source wikipedia* [for more link Manipulators to Wiki!](https://en.wikipedia.org/wiki/Manipulator_(device))

## Robotic Arms
A robotic arm (not robotic hand) is a type of mechanical arm, usually programmable, with similar functions to a human arm; the arm may be the sum total of the mechanism or may be part of a more complex robot. The links of such a manipulator are connected by joints allowing either rotational motion (such as in an articulated robot) or translational (linear) displacement.[1][2] The links of the manipulator can be considered to form a kinematic chain. The terminus of the kinematic chain of the manipulator is called the end effector and it is analogous to the human hand.*source wikipedia*  [for more link to Robotic Arm Wiki!](https://en.wikipedia.org/wiki/Robotic_arm)

## Prerequisite Basic Requiremeent
* ***Python and OOPS in python***
* ***DH Modeling and Analysis of Robotic Arms***
* ***Basics of Matrices***
* ***Python modules***
    * *Numpy*
    * *Matplotlib*
    * *Smympy*
    * *Pytorch*
    * *scikit-learn*

### Methodology
*Create Workspace* ***From Forward Kinematics(FK)*** *which containis ***X Y Z*** *Cordinates and which created from mechanical motion constrains of of* ***Joint Variable Space*** *and using this FK work space Created as input Vector to the* ***Neural network*** *created using Pytorch and* ***Joint Variable Spcae*** *as outputs*.*Thus training the Network* and produce the result.
