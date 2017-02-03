###========================================
#!/bin/bash
#BSUB -n 1 
#BSUB -o lsf.out
#BSUB -e lsf.err
#BSUB -q "windfall"
#BSUB -J module2 
#---------------------------------------------------------------------
#BSUB -R gpu
#BSUB -R "span[ptile=2]"
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/0/output.raw -i ./VectorAdd/Dataset/0/input0.raw,./VectorAdd/Dataset/0/input1.raw -t vector > vectoradd_output0.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/1/output.raw -i ./VectorAdd/Dataset/1/input0.raw,./VectorAdd/Dataset/1/input1.raw -t vector > vectoradd_output1.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/2/output.raw -i ./VectorAdd/Dataset/2/input0.raw,./VectorAdd/Dataset/2/input1.raw -t vector > vectoradd_output2.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/3/output.raw -i ./VectorAdd/Dataset/3/input0.raw,./VectorAdd/Dataset/3/input1.raw -t vector > vectoradd_output3.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/4/output.raw -i ./VectorAdd/Dataset/4/input0.raw,./VectorAdd/Dataset/4/input1.raw -t vector > vectoradd_output4.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/5/output.raw -i ./VectorAdd/Dataset/5/input0.raw,./VectorAdd/Dataset/5/input1.raw -t vector > vectoradd_output5.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/6/output.raw -i ./VectorAdd/Dataset/6/input0.raw,./VectorAdd/Dataset/6/input1.raw -t vector > vectoradd_output6.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/7/output.raw -i ./VectorAdd/Dataset/7/input0.raw,./VectorAdd/Dataset/7/input1.raw -t vector > vectoradd_output7.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/8/output.raw -i ./VectorAdd/Dataset/8/input0.raw,./VectorAdd/Dataset/8/input1.raw -t vector > vectoradd_output8.txt
mpirun -np 1 ./VectorAdd_Solution -e ./VectorAdd/Results/9/output.raw -i ./VectorAdd/Dataset/9/input0.raw,./VectorAdd/Dataset/9/input1.raw -t vector > vectoradd_output9.txt
###end of script
