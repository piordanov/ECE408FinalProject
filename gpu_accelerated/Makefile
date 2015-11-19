COMPILER=nvcc # /usr/local/cuda/bin/nvcc for most machines, but nvcc for EWS
EXENAME1=life_gpu_01
EXENAME2=life_gpu_02
COMPILEFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G # Additional option --std=c++11 can be used on some machines
LINKFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G # Additional option --std=c++11 can be used on some machines
LIBS=-lpng
ARGS=256x256.png output 10

all: $(EXENAME1) $(EXENAME2)

# add additional .o files on the line below (after main_01.o)
$(EXENAME1): main_01.o kernel_01.o pngrw.o utility.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME1) $^ $(LIBS)

$(EXENAME2): main_02.o kernel_02.o pngrw.o utility.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME2) $^ $(LIBS) 

#########################

# add more .cu -> .o compile commands here

kernel_01.o: kernel_01*.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

kernel_02.o: kernel_02*.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main_01.o: main_01*.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main_02.o: main_02*.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cu pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv *.o *output*.png diff.png $(EXENAME1) $(EXENAME2)

run: $(EXENAME1)
	rm -f output*.png diff.png
	./$(EXENAME1) $(ARGS)
	compare output_01.png verify_256x256.png diff.png

run_boundary: $(EXENAME2)
	./$(EXENAME2) $(ARGS)

memcheck: $(EXENAME1)
	rm -f output*.png diff.png
	/usr/local/cuda/bin/cuda-memcheck ./$(EXENAME1) $(ARGS)
	compare output_01.png verify_256x256.png diff.png

valgrind: $(EXENAME1) 
	@echo "Error: Use make -j memcheck instead"
