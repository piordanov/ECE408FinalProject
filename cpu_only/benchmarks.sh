EXENAME=./life_cpu
OUTFILE=benchmarks.txt

echo "" > $OUTFILE

for INFILE in 16x16.png 256x256.png 4096x4096.png
do
    for IX in 1 2 3 4 5
    do
        $EXENAME $INFILE output 256 >> $OUTFILE
        echo "Completed $INFILE trial $IX"
    done
done

