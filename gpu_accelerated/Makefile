COMPILER=/usr/local/cuda/bin/nvcc
EXENAME=life_gpu
COMPILEFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G
LINKFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G
LIBS=-lpng
ARGS=4096x4096.png output 1024

all: $(EXENAME)

# add additional .o files on the line below (after main_01.o)
$(EXENAME): main.o kernel.o pngrw.o utility.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)

#########################

# add more .cu -> .o compile commands here

kernel.o: kernel_01*.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main.o: main_01*.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cu pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv *.o *output*.png diff.png $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)
	rm -f output*.png diff.png
	compare output_01.png verify_256x256.png diff.png

memcheck: $(EXENAME)
	/usr/local/cuda/bin/cuda-memcheck ./$(EXENAME) $(ARGS)
	rm -f output*.png diff.png
	compare output_01.png verify_256x256.png diff.png

valgrind: $(EXENAME) 
	@echo "Error: Use make -j memcheck instead"

