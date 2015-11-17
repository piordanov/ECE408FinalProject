COMPILER=nvcc # /usr/local/cuda/bin/nvcc for most machines, but nvcc for EWS
EXENAME=life_gpu
COMPILEFLAGS=-Xcompiler=-Wall,-Wfatal-errors -G -g
LINKFLAGS=-Xcompiler=-Wall,-Wfatal-errors -G -g
LIBS=-lpng
ARGS=16x16.png output 50

all: $(EXENAME)

# add additional .o files on the line below (after main.o)
$(EXENAME): main.o pngrw.o utility.o kernel.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)

#########################

# add more .cu -> .o compile commands here

kernel.o: kernel.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main.o: main.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cu pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv *.o *output*.png $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

test: $(EXENAME)
	/usr/local/cuda/bin/nvcc/cuda-memcheck ./$(EXENAME) $(ARGS)

valgrind: $(EXENAME) 
	echo "To Laura/Peter - valgrind will throw false positives on cuda or OpenMP code.  I will fix this later.  -Conor."
	valgrind --leak-check=full ./$(EXENAME) $(ARGS)
