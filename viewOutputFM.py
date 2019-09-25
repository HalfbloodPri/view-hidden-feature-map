#将测试数据画出来，看看效果，防止出错
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import time


testDataBoat = np.load('outputFM9_boat.npy')
testDataPlane = np.load('outputFM9_plane.npy')
planeRange = testDataBoat.shape[1]   #二维平面的大小
timeWindow = testDataBoat.shape[0] / 2   #仿真时长
chanels = testDataBoat.shape[3]
timeScale = 1    #时间步长
iterations = int(timeWindow/timeScale)
x,y=np.mgrid[slice(0, planeRange+1, 1),slice(0, planeRange+1, 1)]
levels1 = MaxNLocator(nbins=100).tick_values(testDataPlane.min(), testDataPlane.max())

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('hot')
norm1 = BoundaryNorm(levels1, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im0 = ax0.pcolormesh(x, y, testDataBoat[0,:,:,0], cmap=cmap, norm=norm1)
fig.colorbar(im0, ax=ax0)
ax0.set_title('Boat')

im1 = ax1.pcolormesh(x, y, testDataPlane[0,:,:,0], cmap=cmap, norm=norm1)
fig.colorbar(im1, ax=ax1)
ax1.set_title('Plane')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
def init():
    return im0,im1

chanelNo = 0
def update(iteration):
    im0.set_array(testDataBoat[iteration,:,:,chanelNo].ravel())
    ax0.set_title('Boat '+str(iteration))
    im1.set_array(testDataPlane[iteration,:,:,chanelNo].ravel())
    return im0,im1

for i in range(chanels):
    chanelNo = i
    ani = animation.FuncAnimation(fig, update, frames = range(iterations), interval = 200, init_func=init)
    #plt.show()
    ani.save('testDataGif/%d.gif' % i, fps=5)
