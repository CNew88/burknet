
import numpy as np
import os
import PIL as pl
import burknet as bn

# Data directory name
dirname = './Caleb-Data4/'
attfilename = './Caleb-Data4/LABELS2.csv'

# Target resolution
Ny = 32
Nx = 32

# Zoom
zoomy = 1
zoomx = 1

# Read all attributes
y = np.loadtxt(attfilename, delimiter=',', dtype=str)

ytrain = y[:,1:].astype('uint8')
iexist = y[:,0].astype('int')

exit()
# Image loading and basic formatting
Nc = 3
def load_image(filename):
  img = pl.Image.open(filename)
  pix = np.array(list(img.getdata()))
  arr = np.reshape(pix,[img.height, img.width, pix.shape[1]]).astype('float')
  arr = arr[:, :, 0:Nc]
  return arr

# Read all photos
N = len(iexist)
X = np.zeros((N, Ny, Nx, Nc), dtype='uint8')
for i in range(len(iexist)):
  ifile = iexist[i]
  
  # Load images    
  arr = load_image(os.path.join(dirname, ("%d" % ifile) + '.png'))
  arr_ice = load_image(os.path.join(dirname, ("%d" % ifile) + '_ice.png'))

  # Do these only once since they don't change between pictures
  if ifile == 0:
    thres = 10
    ygrid, xgrid, cgrid = np.mgrid[0:arr.shape[0], 0:arr.shape[1], 0:arr.shape[2]]
    cy = arr.shape[0]//2
    cx = arr.shape[1]//2
  
  # Detect center  
  y_center = int(np.round(np.mean(ygrid[np.abs(arr-arr_ice) > thres])))
  x_center = int(np.round(np.mean(xgrid[np.abs(arr-arr_ice) > thres])))

  # Re-center image
  arr_centered = np.roll(arr, shift=(cy-y_center, cx-x_center, 0), axis=(0, 1, 2))

  # Do these only once since they don't change between pictures
  if ifile == 0:
    ys, xs, cs= np.mgrid[0:Ny, 0:Nx, 0:Nc]
    ysc = cy + (ys - Ny/2) / zoomy  # center coordinates around image center
    xsc = cx + (xs - Nx/2) / zoomx
    y1 = np.floor(ysc).astype('int')
    ky = ysc - y1
    x1 = np.floor(xsc).astype('int')
    kx = xsc - x1

  # Clip image
  img = (1 - kx) * ((1 - ky) * arr_centered[y1, x1,     cs] + ky * arr_centered[y1 + 1, x1,     cs]) +\
              kx * ((1 - ky) * arr_centered[y1, x1 + 1, cs] + ky * arr_centered[y1 + 1, x1 + 1, cs])

  # Place in database
  X[i, :, :, :] = img.astype('uint8')

# Randomize data array  
np.random.shuffle(X)

# Create data object
data = bn.Data()
data.create(X, ytrain, 'N/A', 'N/A', 'N/A', 'N/A')

# Save data object
savefile = './data/caleb4 Ny' + str(Ny) + ' Nx' + str(Nx) + ' zoomy' + str(zoomy) + ' zoomx' + str(zoomx) + ' Ntrain' + str(X.shape[0]) + '.pkl.gz'
data.save(savefile)
