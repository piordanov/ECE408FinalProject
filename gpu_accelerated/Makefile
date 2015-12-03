COMPILER=/usr/local/cuda/bin/nvcc # for most machines, but nvcc for EWS
EXENAME=life_gpu
COMPILEFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G --std=c++11
LINKFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G --std=c++11
LIBS=-lpng
ARGS=680x384.png output 256

all: $(EXENAME)

# add additional .o files on the line below (after main_01.o)
$(EXENAME): main.o kernel.o pngrw.o utility.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS) 

#########################

# add more .cu -> .o compile commands here

kernel.o: kernel_08*.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main.o: main_05*.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cu pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv *.o *output*.png diff.png $(EXENAME)

run: $(EXENAME)
	rm -f output*.png diff.png
	./$(EXENAME) $(ARGS)
#	compare output_001.png verify_256x256.png diff.png

memcheck: $(EXENAME)
	rm -f output*.png diff.png
	/usr/local/cuda/bin/cuda-memcheck ./$(EXENAME) $(ARGS)
#	compare output_001.png verify_256x256.png diff.png

valgrind: $(EXENAME) 
	@echo "Error: Use make -j memcheck instead"

