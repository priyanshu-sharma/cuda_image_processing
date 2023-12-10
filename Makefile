
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3 `pkg-config --libs --cflags opencv`
endif
LD_FLAGS    = -lcudart `pkg-config --libs --cflags opencv`
EXE	        = starter
OBJ	        = starter.o

default: $(EXE)

starter.o: starter.cu kernel.cu
	$(NVCC) -c -o $@ starter.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)