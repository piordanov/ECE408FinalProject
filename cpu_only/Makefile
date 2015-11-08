COMPILER=g++
EXENAME=life_cpu
COMPILEFLAGS=-Wall -Wfatal-errors -g
LINKFLAGS=-Wall -Wfatal-errors -g
LIBS=-lpng
ARGS=test_input_01.png 12

all: $(EXENAME)

# add additional .o files after main.o
$(EXENAME): main.o pngrw.o recalculate_grid_cpu.o timing.o
	$(COMPILER) $(LINKFLAGS) -o $(EXENAME) $^ $(LIBS)

#########################

# add more .cpp -> .o compile commands here

main.o: main.cpp ece408_final_cpu.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

pngrw.o: pngrw.cpp pngrw.h
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

recalculate_grid_cpu.o: recalculate_grid_cpu.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

timing.o: timing.cpp
	$(COMPILER) $(COMPILEFLAGS) -c -o $@ $<

#########################

clean:
	rm -fv generation*.png *.o $(EXENAME)

run: $(EXENAME)
	./$(EXENAME) $(ARGS)

valgrind: $(EXENAME)
	valgrind --leak-check=full ./$(EXENAME) $(ARGS)

