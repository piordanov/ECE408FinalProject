COMPILER=nvcc # /usr/local/cuda/bin/nvcc for most machines, but nvcc for EWS
EXENAME1=life_gpu_01
EXENAME2=life_gpu_02
COMPILEFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G
LINKFLAGS=-Xcompiler=-Wall,-Wfatal-errors -g -G
LIBS=-lpng
ARGS=4096x4096.png output 10

all: $(EXENAME1) $(EXENAME2)

# add additional .o files on the line below (after main_01.o)
$(EXENAME1): main_01.o pngrw.o utility.o kernel_01.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME1) $^ $(LIBS)

$(EXENAME2): main_02.o pngrw.o utility.o kernel_02.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME2) $^ $(LIBS)

#########################

# add more .cu -> .o compile commands here

kernel_01.o: kernel_01_vanilla.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

kernel_02.o: kernel_02_cpu_boundary.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main_01.o: main_01_vanilla.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

main_02.o: main_02_cpu_boundary.cu pngrw.h ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cu pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cu ece408_final_gpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv *.o *output*.png $(EXENAME1) $(EXENAME2)

run_vanilla: $(EXENAME1)
	./$(EXENAME1) $(ARGS)

run_boundary: $(EXENAME2)
	./$(EXENAME2) $(ARGS)

test_vanilla: $(EXENAME1)
	/usr/local/cuda/bin/nvcc/cuda-memcheck ./$(EXENAME1) $(ARGS)

valgrind_vanilla: $(EXENAME1)
	@echo "Error: Use make -j memcheck instead"
