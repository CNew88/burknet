
import numpy as np
import os
import PIL as pl
import burknet as bn

# Data directory name
dirname = './rawdata/celeba/img_align_celeba/'
attfilename = './rawdata/celeba/list_attr_celeba.txt'

# Target resolution
Ny = 80
Nx = 64

# Zoom
zoom = .6

# Read all attributes
y = np.loadtxt(attfilename, skiprows=2, dtype=str)
ytrain = (y[:,1:].astype('uint8') + 1)//2

# Read all photos
N = 202599
Nc = 3
X = np.zeros((N, Ny, Nx, Nc), dtype='uint8')
for ifile in range(N):
  filename = ("%06d" % (ifile+1)) + '.jpg'
  fullname = os.path.join(dirname, filename)
  imgpl = pl.Image.open(fullname)
  pix = np.array(list(imgpl.getdata()))
  arr = np.reshape(pix,[imgpl.height, imgpl.width, pix.shape[1]]).astype('float')

  # Do these only once, since they don't change between pictures
  if ifile == 0:
    # Original image coord
    ygrid, xgrid, cgrid = np.mgrid[0:arr.shape[0], 0:arr.shape[1], 0:arr.shape[2]]
    cy = arr.shape[0]//2
    cx = arr.shape[1]//2
  
    # Clipped coord
    ys, xs, cs= np.mgrid[0:Ny, 0:Nx, 0:Nc]
    ysc = cy + (ys - Ny/2) / zoom  # center coordinates around image center
    xsc = cx + (xs - Nx/2) / zoom
    y1 = np.floor(ysc).astype('int')
    ky = ysc - y1
    x1 = np.floor(xsc).astype('int')
    kx = xsc - x1

    # Clip image
  img = (1 - kx) * ((1 - ky) * arr[y1, x1,     cs] + ky * arr[y1 + 1, x1,     cs]) +\
              kx * ((1 - ky) * arr[y1, x1 + 1, cs] + ky * arr[y1 + 1, x1 + 1, cs])

  if ifile % 10000 == 0:
    print("%d" % ifile + " of " + "%d" % N + " is complete...")

  # Place in database
  X[ifile, :, :, :] = img

# Randomize data array  
np.random.shuffle(X)

# Create data object
data = bn.Data()
data.create(X, ytrain, 'N/A', 'N/A', 'N/A', 'N/A')

# Save data object
savefile = './data/celeba Ny' + str(Ny) + ' Nx' + str(Nx) + ' zoom' + str(zoom) + ' Ntrain' + str(X.shape[0]) + '.pkl.gz'
data.save(savefile)
