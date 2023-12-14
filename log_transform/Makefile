
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3 `pkg-config --libs --cflags opencv`
endif
LD_FLAGS    = -lcudart `pkg-config --libs --cflags opencv`
EXE	        = log_transformation
OBJ	        = log_transformation.o

default: $(EXE)

log_transformation.o: log_transformation.cu kernel.cu
	$(NVCC) -c -o $@ log_transformation.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) *.jpeg