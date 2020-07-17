# circle-tensordump

_circle-tensordump_ allows users to retrieve tensor information from a Circle model file

## options

**--tensors**

dump tensors in circle file

```
$ ./circle-tensordump --tensors ../luci/tests/Conv2D_000.circle
                      
----------------------------------------------------------------------
[ifm]
 └── shape : (1, 3, 3, 2)

----------------------------------------------------------------------
[ker]
 ├── shape : (1, 1, 1, 2)
 └── buffer
     ├── index : 3
     ├── size  : 8
     └── data  : 0.727939, 0.320132, 

----------------------------------------------------------------------
[bias]
 ├── shape : (1)
 └── buffer
     ├── index : 4
     ├── size  : 4
     └── data  : -0.794465, 

----------------------------------------------------------------------
[ofm]
 └── shape : (1, 3, 3, 1)
```

**--tensors_to_hdf5**

dump tensors in circle file to hdf5 file

```
$ ./circle-tensordump --tensors_to_hdf5 ../luci/tests/Conv2D_000.circle output_path.h5
$ h5dump output_path.h5

HDF5 "output_path.h5" {
GROUP "/" {
   GROUP "bias" {
      DATASET "weights" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
         DATA {
         (0): -0.794465
         }
      }
   }
   GROUP "ifm" {
   }
   GROUP "ker" {
      DATASET "weights" {
         DATATYPE  H5T_IEEE_F32LE
         DATASPACE  SIMPLE { ( 1, 1, 1, 2 ) / ( 1, 1, 1, 2 ) }
         DATA {
         (0,0,0,0): 0.727939, 0.320132
         }
      }
   }
   GROUP "ofm" {
   }
}
}
```
