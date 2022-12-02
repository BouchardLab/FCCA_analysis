#!/bin/bash
mpirun -n 8 python s1m1vt.py 0 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 1 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 2 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 3 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 4 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 5 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 6 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
mpirun -n 8 python s1m1vt.py 7 0 --error_thresh=1.00 --error_op=le --q=0.00 --filter_op=le
