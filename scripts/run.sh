for i in $(seq 6 42);
do
    python3 handover_data_visualization.py -t "plate" -d $i
done
