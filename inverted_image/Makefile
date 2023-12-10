
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3 `pkg-config --libs --cflags opencv`
endif
LD_FLAGS    = -lcudart `pkg-config --libs --cflags opencv`
EXE	        = inverted_image
OBJ	        = inverted_image.o

default: $(EXE)

inverted_image.o: inverted_image.cu kernel.cu
	$(NVCC) -c -o $@ inverted_image.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)