EXENAME=./life_gpu
OUTFILE=benchmarks.txt

echo "" > $OUTFILE

for INFILE in \
    256x256_dense.png \
    512x512_dense.png \
    1024x1024_dense.png \
    2048x2048_dense.png \
    4096x4096_dense.png \
    8192x8192_dense.png \
    256x256_sparse.png \
    512x512_sparse.png \
    1024x1024_sparse.png \
    2048x2048_sparse.png \
    4096x4096_sparse.png \
    8192x8192_sparse.png
do
    for IX in 1 2 3 4 5
    do
        $EXENAME $INFILE output 256 >> $OUTFILE
        echo "Completed $INFILE trial $IX"
    done
done

