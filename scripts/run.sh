for i in $(seq 1 42);
do
    python3 data_processing_bimanual.py -t plate -d $i
done