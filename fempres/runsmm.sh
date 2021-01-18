#!/bin/bash
cd $HOME
cd Fempres
cd fempres  
module load python3/3.7.4
module load openmpi/4.0.2

mpiexec -n 3840  python3 -m mpi4py smm.py
 

