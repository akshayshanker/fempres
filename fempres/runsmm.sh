#!/bin/bash
cd $HOME
cd Fempres
cd fempres  
module load python3/3.7.4
module load openmpi/4.0.2


for var in {1..10}
do
	mpiexec -n 19680  python3 -m mpi4py smm.py
done

 

