
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 
else
NVCC_FLAGS  = -O3 `pkg-config --libs --cflags opencv`
endif
LD_FLAGS    = -lcudart `pkg-config --libs --cflags opencv`
EXE	        = image_thresholding
OBJ	        = image_thresholding.o

default: $(EXE)

image_thresholding.o: image_thresholding.cu kernel.cu
	$(NVCC) -c -o $@ image_thresholding.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) *.jpeg