"""
Simulation for slime behavior
@author: Zhangyiyuan

If you find the code in this repository useful for your research consider citing it:
@misc{Simulation_slime_behavior_2019,
      author = {Zhangyiyuan},
      title = {Simulation for slime behavior},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/sdgds/Simulation_slime_behavior}}
      }
"""


## Import some packages needed
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


## Define some useful function
# caculate euchlid distance between two points
def Euchlid_dict(point_a, point_b):
    '''
    point_a[array]: 2D point axis
    point_b[array]: 2D point axis
    '''
    return np.sqrt((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)

# def rotation matrix by location_now and location_old (Markov process)
def Rotation_matrix(location_change):
    '''
    location_change[array]: change between location_now and location_old
    '''
    if location_change[0]!=0:
        tan_angle = location_change[1]/location_change[0]
        angle = math.atan(tan_angle)
    else:
        angle = np.random.uniform(-np.pi, np.pi, 1).item()
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

# the sampling behavior of slime iid to Gauss distribution       
def Gauss_2d(mu, sigma): 
    '''
    mu[array]: mean
    sigma[array]: std
    '''
    x = np.random.normal(mu[0],sigma[0],100)
    y = np.random.normal(mu[1],sigma[1],100)
    s = np.vstack((x,y)).T
    return s
mu = np.array([0,0])
sigma = np.array([1,0.01])
s = Gauss_2d(mu,sigma)
#plt.figure(figsize=(6,6))
#plt.xlim((-6,6))
#plt.ylim((-6,6))
#plt.plot(s[:,0],s[:,1],'*')

# def update rule
def Rule_update(location_old, location_now, food_location, reward_factor):
    '''
    location_old[array]
    location_now[array]
    food_location[array]
    reward_factor[int]: the weight of reward for every step
    '''
    if (location_now!=location_old).all():
        location_change = np.array([location_now[0]-location_old[0], 
                                    location_now[1]-location_old[1]])
        rotation_matrix = Rotation_matrix(location_change)
        
        d_old = Euchlid_dict(location_old, food_location)
        reward_old = 1/d_old
        d_now = Euchlid_dict(location_now, food_location)
        reward_now = 1/d_now
        reward_change = reward_now - reward_old
        
        s = Gauss_2d(mu, sigma)
        s[:,0] = s[:,0] + reward_factor*reward_change
        for i in range(100):
            s[i] = np.dot(rotation_matrix, s[i])
        location_new = random.sample(list(s), 1)[0]
        return location_now, location_new
    if (location_now==location_old).all():
        location_now = location_old
        location_change = np.array([location_now[0]-location_old[0], 
                                    location_now[1]-location_old[1]])
        rotation_matrix = Rotation_matrix(location_change)
        
        d_old = Euchlid_dict(location_old, food_location)
        reward_old = 1/d_old
        d_now = Euchlid_dict(location_now, food_location)
        reward_now = 1/d_now
        reward_change = reward_now - reward_old
        
        s = Gauss_2d(mu, sigma)
        s[:,0] = s[:,0] + reward_factor*reward_change
        for i in range(100):
            s[i] = np.dot(rotation_matrix, s[i])
        location_new = random.sample(list(s), 1)[0]
        return location_now, location_new
    
    

## Iteration of slime behavior
# start point
x_start = 0
y_start = 0
start_location = np.array([x_start, y_start])

# food location
food_location = np.array([0,5])

# iteration
Location = []
Location.append(start_location)
l1 = start_location
l2 = start_location
D_start = []
D_food = []
for step in range(1000):
    l1,l2 = Rule_update(l1, l2, food_location, reward_factor=10)
    d_start = Euchlid_dict(l2, start_location)
    D_start.append(d_start)
    d_food = Euchlid_dict(l2, food_location)
    D_food.append(d_food)
    if d_start >= 5:
        l2 = l1
        Location.append(l2)
    if d_start < 5:
        Location.append(l2)
    if d_food < 0.5:
        break
Location = np.array(Location)

# plot the slime behavior
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.axis('off')
plt.xlim((-5,5))
plt.ylim((-5,5))
circle = Circle(xy=(0,0), radius=5, alpha=0.5)
plt.plot([0], [5], 'ro-')
plt.plot(Location[:,0], Location[:,1], 'y-')
ax.add_patch(circle)
plt.show()

print('The max distance to start point', max(D_start))
print('The min distance to food point', min(D_food))


