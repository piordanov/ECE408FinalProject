COMPILER=g++
EXENAME=life_cpu
COMPILEFLAGS=-Wall -Wfatal-errors -g -fopenmp
LINKFLAGS=-Wall -Wfatal-errors -g -fopenmp
LIBS=-lpng
ARGS=256x256.png output 1

all: $(EXENAME)

# add additional .o files after main.o
$(EXENAME): main.o pngrw.o recalculate_grid_cpu.o utility.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)

#########################

# add more .cpp -> .o compile commands here

main.o: main.cpp ece408_final_cpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cpp pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

recalculate_grid_cpu.o: recalculate_grid_cpu_03*.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

utility.o: utility.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv output*.png *.o $(EXENAME)

run: $(EXENAME)
	rm -rf output*.png
	./$(EXENAME) $(ARGS)

valgrind: $(EXENAME)
	rm -rf output*.png
	valgrind --leak-check=full ./$(EXENAME) $(ARGS)

