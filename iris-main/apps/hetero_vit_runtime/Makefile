include ../makefile_defs.mk

all: test_cost_model libPolicyHeteroEFT.so kernel.openmp.so vit_dag_skeleton

test_cost_model: test_cost_model.cpp hetero_cost_model.h
	$(CXX) -std=c++11 -O2 -o $@ test_cost_model.cpp

libPolicyHeteroEFT.so: PolicyHeteroEFT.cpp hetero_cost_model.h
	$(CXX) $(CXXFLAGS) -std=c++11 -fPIC -shared -o $@ PolicyHeteroEFT.cpp

kernel.openmp.so: kernel.openmp.c
	$(CC) $(CFLAGS) -O3 -fopenmp -fPIC -shared -o $@ kernel.openmp.c

vit_dag_skeleton: vit_dag_skeleton.c vit_task_manifest.h hetero_cost_model.h
	$(CC) $(CFLAGS) -o $@ vit_dag_skeleton.c $(LDFLAGS)

clean:
	rm -f test_cost_model libPolicyHeteroEFT.so kernel.openmp.so vit_dag_skeleton
