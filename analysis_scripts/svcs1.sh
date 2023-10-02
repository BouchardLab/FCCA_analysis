#!/bin/bash
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 0
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 1
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 2
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 3
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 4
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 5
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 6
mpirun -n 8 python decodingvt_cv_strvcorrS1.py 7
