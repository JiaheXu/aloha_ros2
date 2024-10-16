for i in $(seq 1 5);
do
    python3 calib_verify.py -t pick_up_plate -d $i
done
