#!/usr/bin/env python

import sys
import obspy
from os.path import exists

import numpy as np
import pylab


def read_fortran(filename):
	with open(filename, 'rb') as f:
		f.seek(0)
		v = np.fromfile(f, dtype='float32')

	return v[:-1]

if __name__ == '__main__':
	""" Plots su(Seismic Unix) data

	  SYNTAX
		  plot_trace  folder_name  component_name||file_name  (source id)
		  e.g. ./plot_trace.py output vx
		       ./plot_trace.py output vx 0
		       ./plot_trace.py output vx_000000.su
	"""
	istr = ''
	if len(sys.argv) > 3:
		istr = str(sys.argv[3])
		while len(istr) < 6:
			istr = '0' + istr
	else:
		istr = '000000'

	path = "%s/%s" % (sys.argv[1], sys.argv[2])
	if not exists(path):
		path = "%s/%s.su" % (sys.argv[1], sys.argv[2])
	if not exists(path):
		path = '%s/%s_%s.su' % (sys.argv[1], sys.argv[2], istr)

	assert exists(path)

	data = obspy.read(path, format='SU', byteorder='<')
	stats = data[0].stats
	t = np.arange(stats.npts)*stats.delta
	am = 0

	print('dt    = ', stats.delta)
	print('npts  = ', stats.npts)
	print('nsrc  = ', len(data))

	for i in range(len(data)):
		am = max(am, np.amax(data[i].data))

	for i in range(len(data)):
		pylab.plot(t, data[i].data + i * am, 'b')

	print('amax  = ', am)
	pylab.show()
