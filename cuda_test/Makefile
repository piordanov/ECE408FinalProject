COMPILER=/usr/local/cuda-7.0/bin/nvcc # on most machines, but not EWS machines
EXENAME=cuda_test
COMPILEFLAGS=-Xcompiler -Wall
LINKFLAGS=-Xcompiler -Wall
LIBS=
ARGS=

all: $(EXENAME)

$(EXENAME): cuda_test.cu
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)

#########################

clean:
	rm -fv $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

test: $(EXENAME)
	cuda-memcheck ./$(EXENAME) $(ARGS)

