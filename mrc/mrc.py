"""Read and write MRC2014 files.

This is a simple module to read and write images in the MRC2014. MRC
files store a single 3D array, usually interpreted as either

  #. a stack of 2D images, with the third array index specifying the
    image number in the stack
  #. a 3D volume.

:Example:

Load file into an ndarray 

>>> import MRC
>>> image_data = MRC.read_MRC( 'filename.mrc' )

Write ndarray to file, using defaults for metadata

>>> import MRC
>>> MRC.write_MRC( image_data )

Load file and return an MRCFile object (ndarray + metadata)

>>> from MRC import MRCFile
>>> image_file = MRCFile( 'filename.mrc' )
>>> image_file.header

Write ndarray and metadata to file

>>> from MRC import MRCFile
>>> image_file = MRCFile( image_data )
>>> image_file.header.map_indices = (2,1,3)
>>> image_file.write_MRC()
"""

"""Copyright (c) 2016, David Dynerman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

#from __future__ import print_function

import sys
import struct
import timeit
import scipy as sp
import numpy as np
import os.path

from collections import namedtuple

# MRC2014 header specification
def _is_positive(x):
    return x > 0

def _is_non_negative(x):
    return x >= 0

class MRCFile:
    _mrc_header_specification = [ [0, 'I', "NX",
                                   """number of columns 
                                   (fastest changing in map)""",
                                   _is_positive ],
                                  [4, 'I', "NY", "number of rows",
                                   _is_positive ],
                                  [8, 'I', "NZ",
                                   """number of sections 
                                   (slowest changing in map)""",
                                   _is_positive ],
                                  [12, 'I', "MODE",
                                   """0 8-bit signed integer (range -128 to 127)
                                      1 16-bit signed integer
                                      2 32-bit signed real
                                      3 transform : complex 16-bit integers
                                      4 transform : complex 32-bit reals
                                      6 16-bit unsigned integer""",
                                   lambda mode: mode in [0, 1, 2, 3, 4, 6] ],
                                  [16, 'I', "NXSTART",
                                   """number of first column in map 
                                   (Default = 0)""",
                                   _is_non_negative ],
                                  [20, 'I', "NYSTART",
                                   "number of first row in map",
                                   _is_non_negative ],
                                  [24, 'I', "NZSTART",
                                   "number of first section in map",
                                   _is_non_negative ],
                                  [28, 'I', "MX",
                                   "number of intervals along X",
                                   _is_non_negative ],
                                  [32, 'I', "MY",
                                   "number of intervals along Y",
                                   _is_non_negative ],
                                  [36, 'I', "MZ",
                                   "number of intervals along Z",
                                   _is_non_negative ],
                                  [40, '3f', "CELLA",
                                   "cell dimensions in angstroms"],
                                  [52, '3f', "CELLB", "cell angles in degrees"],
                                  [64, 'I', "MAPC",
                                   "axis corresp to cols (1,2,3 for X,Y,Z)",
                                   lambda mapc: mapc in [1,2,3] ],
                                  [68, 'I', "MAPR",
                                   "axis corresp to rows (1,2,3 for X,Y,Z)",
                                   lambda mapr: mapr in [1,2,3] ],
                                  [72, 'I', "MAPS",
                                   """axis corresp to sections 
                                      (1,2,3 for X,Y,Z)""",
                                   lambda maps: maps in [1,2,3] ],
                                  [76, 'f', "DMIN", "minimum density value"],
                                  [80, 'f', "DMAX", "maximum density value"],
                                  [84, 'f', "DMEAN", "mean density value"],
                                  [88, 'I', "ISPG", "space group number",
                                   _is_positive ],
                                  [92, 'I', "NSYMBT",
                                   """size of extended header 
                                      (which follows main header) in bytes""",
                                   _is_non_negative ],
                                  [96, '100s', "EXTRA",
                                   """extra space used for anything - 
                                      0 by default"""],
                                  [104, 'I', "EXTTYP",
                                   "code for the type of extended header"],
                                  [108, 'I', "NVERSION",
                                   "version of the MRC format"],
                                  [196, '3f', "ORIGIN",
                                   """phase origin (pixels) or origin of 
                                   subvolume (A)"""],
                                  [208, '4s', "MAP",
                                   """character string 'MAP ' to identify 
                                      file type""",
                                   lambda mapstr: mapstr[0:3] == b'MAP' ],
                                  [212, '4s', "MACHST",
                                   """machine stamp encoding byte ordering 
                                      of data"""],
                                  [216, 'f', "RMS",
                                   "rms deviation of map from mean density"],
                                  [220, 'I', "NLABL",
                                   "number of labels being used",
                                   _is_non_negative ],
                                  [224, '800s', "LABEL",
                                   "10 80-character text labels"] ]

    MRCHeader = namedtuple( 'MRCHeader', [ header_entry[2] for header_entry in
                                           _mrc_header_specification ] )
    def __init__(self, *args):
        if len(args) is not 1:
            raise ValueError('Expected exactly one argument')

        if isinstance(args[0], str):
            self._from_file(args[0])
        else:
            self._from_array(args[0])

    def _from_file(self, filename,prefetch=False):
        self.mrc_fd = open(filename, 'rb')

        last_header_entry = max(_mrc_header_specification, key=lambda x: x[0])
        
        header_byte_size = last_header_entry[0] + struct.calcsize(last_header_entry[1])

        # MRC2014 file header should always be 1024 bytes
        assert header_byte_size == 1024, """MRC2014 header size must be 1024
        bytes. Check for an error in _mrc_header_specification"""
        
        self.mrc_fd = open(filename, 'rb')
        header_raw = self.mrc_fd.read(header_byte_size)

        # TODO: Document how this works, probably up in the header
        # specification
        header_fields = [ struct.unpack(header_entry[1],
                                        header_raw[header_entry[0]:header_entry[0]+struct.calcsize(header_entry[1])])
                          for header_entry in
                          _mrc_header_specification ]

        # flatten length 1 tulpes in header
        header_args = [ header_entry[0] if len(header_entry) == 1 else
        header_entry for header_entry in header_fields ]

        self.header = MRCHeader(*header_args)
        
        # Check if header contains correct values
        for header_entry in _mrc_header_specification:
            if len(header_entry) == 5:
                header_value = getattr(header, header_entry[2])
                if not header_entry[4](header_value):
                    log.error(header_entry[3])
                    raise ValueError("""MRC file header in {} contained an
                    invalid value for {} - is the file
                    corrupted?""".format(filename, header_entry[2]))

        data_shape = (self.header.NZ, self.header.NY, self.header.NX)
        
        if self.header.MODE == 0:
            self._bytes_per_section = 1*self.header.NX*self.header.NY
            dt = np.int8
        elif self.header.MODE == 1:
            self._bytes_per_section = 2*self.header.NX*self.header.NY
            dt = np.int16
        elif self.header.MODE == 2:
            self._bytes_per_section = 4*self.header.NX*self.header.NY
            dt = np.float32
        elif self.header.MODE == 3:
            raise NotImplementedError("MRC Mode 3 data not implemented")
        elif self.header.MODE == 4:
            raise NotImplementedError("MRC Mode 4 data not implemented")            
        elif self.header.MODE == 6:
            self._bytes_per_section = 2*self.header.NX*self.header.NY
            dt = np.uint16

        self.mrc_fd.seek(header_byte_size)
            
        if self.header.NSYMBT > 0:
            self.extra_header_data = struct.unpack(
                '{}s'.format(self.header.NSYMBT),
                self.mrc_fd.read(self.header.NSYMBT))
            
        self.mrc_fd.seek(header_byte_size + self.header.NSYMBT)
            
        self.data = np.frombuffer(self.mrc_fd.read(-1),
                                  dtype=dt).reshape(self.header.NZ,
                                                    self.header.NY,
                                                    self.header.NX)
        self.mrc_fd.close()
        
    def __getitem__(self,index):
        if len(index) == 2:
            index = (0,) + index

        if len(index) is not 3:
            raise ValueError("Expected 2 or 3 indices")
            
        return self.data[index]

    def write_file(self, filename):
        new_mrc_fd = open(filename, 'wb')

        for header_entry in _mrc_header_specification:
            new_mrc_fd.seek(header_entry[0])
            new_mrc_fd.write(struct.pack(header_entry[1],
                                         getattr(self.header,
                                                 header_entry[2])))
        
        last_header_entry = max(_mrc_header_specification, key=lambda x: x[0])
        header_byte_size = last_header_entry[0] + struct.calcsize(last_header_entry[1])
        new_mrc_fd.seek(header_byte_size)

        if self.header.NSYMBT > 0:
            new_mrc_fd.write(struct.pack('{}s'.format(self.header.NSYMBT),
            self.extra_header_data))

        new_mrc_fd.seek(header_byte_size + self.header.NSYMBT)

        self.data.flatten().tofile(new_mrc_fd)

        new_mrc_fd.close()
        

