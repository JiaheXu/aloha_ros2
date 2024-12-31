for i in $(seq 1 25);
do
    # python3 data_processing_bimanual_trajectory.py -t close_marker -d $i 
    python3 data_processing_bimanual_keypose.py -t insert_battery -d $i -debug 1
    # python3 data_processing_bimanual_trajectory.py -t insert_battery -d $i 
done


# easy: 
# lift_ball # huge error on EE
# stack blocks



# hard 
# close pen
# hand_over block
# ziploc # huge error on EE
# 
