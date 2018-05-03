#!/usr/bin/env python

import sys
from os.path import exists

import numpy as np
import pylab
import scipy.interpolate


def read_fortran(filename):
	""" Reads Fortran style binary data and returns a numpy array.
	"""
	with open(filename, 'rb') as f:
		# read size of record
		f.seek(0)
		n = np.fromfile(f, dtype='int32', count=1)[0]

		# read contents of record
		f.seek(4)
		v = np.fromfile(f, dtype='float32')

	return v[:-1]


def mesh2grid(v, x, z):
	""" Interpolates from an unstructured coordinates (mesh) to a structured
		coordinates (grid)
	"""
	lx = x.max() - x.min()
	lz = z.max() - z.min()
	nn = v.size
	mesh = _stack(x, z)

	nx = np.around(np.sqrt(nn*lx/lz))
	nz = np.around(np.sqrt(nn*lz/lx))
	dx = lx/nx
	dz = lz/nz

	# construct structured grid
	x = np.linspace(x.min(), x.max(), nx)
	z = np.linspace(z.min(), z.max(), nz)
	X, Z = np.meshgrid(x, z)
	grid = _stack(X.flatten(), Z.flatten())

	# interpolate to structured grid
	V = scipy.interpolate.griddata(mesh, v, grid, 'linear')

	# workaround edge issues
	if np.any(np.isnan(V)):
		W = scipy.interpolate.griddata(mesh, v, grid, 'nearest')
		for i in np.where(np.isnan(V)):
			V[i] = W[i]

	return np.reshape(V, (int(nz), int(nx))), x, z



def _stack(*args):
	return np.column_stack(args)



if __name__ == '__main__':
	""" Plots data on 2-D unstructured mesh
	  Modified from a script for specfem2d:
	  http://tigress-web.princeton.edu/~rmodrak/visualize/plot2d

	  Can be used to plot models or kernels created by inv.cu

	  SYNTAX
		  plot_model  folder_name  component_name||file_name  (time_step)
		  e.g. ./plot_model.py output vx 1000
		       ./plot_model.py output proc001000_vx.bin 
		       ./plot_model.py example/model/checker vs
	"""

	istr = ''
	if len(sys.argv) > 3:
		istr = str(sys.argv[3])
		while len(istr) < 6:
			istr = '0' + istr
	else:
		istr = '000000'

	# parse command line arguments
	x_coords_file = '%s/proc000000_x.bin' % sys.argv[1]
	z_coords_file = '%s/proc000000_z.bin' % sys.argv[1]

	# check that files actually exist
	assert exists(x_coords_file)
	assert exists(z_coords_file)

	database_file = "%s/%s" % (sys.argv[1], sys.argv[2])
	if not exists(database_file):
		database_file = "%s/%s.bin" % (sys.argv[1], sys.argv[2])
	if not exists(database_file):
		database_file = "%s/proc%s_%s.bin" % (sys.argv[1], istr, sys.argv[2])

	assert exists(database_file)

	# read mesh coordinates
	#try:
	if True:
		x = read_fortran(x_coords_file)
		z = read_fortran(z_coords_file)
	#except:
	#    raise Exception('Error reading mesh coordinates.')

	# read database file
	try:
		v = read_fortran(database_file)
	except:
		raise Exception('Error reading database file: %s' % database_file)

	# check mesh dimensions
	assert x.shape == z.shape == v.shape, 'Inconsistent mesh dimensions.'


	# interpolate to uniform rectangular grid
	V, X, Z = mesh2grid(v, x, z)

	# display figure
	pylab.pcolor(X, Z, V)
	locs = np.arange(X.min(), X.max() + 1, (X.max() - X.min()) / 5)
	pylab.xticks(locs, map(lambda x: "%g" % x, locs / 1e3))
	locs = np.arange(Z.min(), Z.max() + 1, (Z.max() - Z.min()) / 5)
	pylab.yticks(locs, map(lambda x: "%g" % x, locs / 1e3))
	pylab.colorbar()
	pylab.xlabel('x / km')
	pylab.ylabel('z / km')
	pylab.gca().invert_yaxis()
	pylab.show()
