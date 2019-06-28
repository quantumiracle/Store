import pybullet as p
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

physicsClient = p.connect(p.GUI)

# p.setGravity(0, 0, -0.001)
planeId = p.loadURDF("plane.urdf")
bunnyId = p.loadSoftBody("bunny.obj")
#ballId = p.loadSoftBody("softball.obj")
# filmId = p.loadSoftBody("film.obj")
# squareId = p.loadSoftBody("square.obj")
# cylinder = p.loadMJCF("mjcf/cylinder_fromtoZ.xml")
# reacher = p.loadMJCF("mjcf/reacher3.xml")
# squareId = p.loadSoftBody("cloth.obj", basePosition = [0,0,10], baseOrientation=p.getQuaternionFromEuler([90,0,0]))
squareId = p.loadSoftBody("cloth2.obj", basePosition = [0,3,5], baseOrientation=p.getQuaternionFromEuler([90,0,0]))
# tacId = p.loadSoftBody("tac2obj.obj", basePosition = [0,3,5], baseOrientation=p.getQuaternionFromEuler([90,0,0]))
# tacId = p.loadSoftBody("tac1.obj", basePosition = [0,3,5], baseOrientation=p.getQuaternionFromEuler([90,0,0]))

# a different property cloth with more elasticity, but almost the same
object_position = [0,3,5]
# squareId = p.loadSoftBody("cloth3.obj", basePosition = object_position, baseOrientation=p.getQuaternionFromEuler([90,0,0]))  

# for i in range (10):
  # p.loadURDF("cube_small.urdf", [0, 0, 0.2*i])

cubeID=p.loadURDF("cube_small.urdf", [0, 0, 1])
useRealTimeSimulation = 1

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)
i=0
while p.isConnected():
  i+=1
  print(p.getBasePositionAndOrientation(cubeID))
  # p.resetBasePositionAndOrientation(cubeID, [np.cos(i/360*2*np.pi), np.sin(i/360*2*np.pi), 1], p.getQuaternionFromEuler([0,0,0]))

  p.setGravity(0, 0, -10)
  if (useRealTimeSimulation):

    sleep(0.01)  # Time in seconds.
    camera_position = object_position
    camera_position[1]+=3
    viewMatrix = p.computeViewMatrix(camera_position, object_position, [0,1,0])
    img = p.getCameraImage(320,200,viewMatrix=viewMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL )  # seems not work yet
    plt.imshow(img[2])
    # plt.show()
  else:
    p.stepSimulation()
