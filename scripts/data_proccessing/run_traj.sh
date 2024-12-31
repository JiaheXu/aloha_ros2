# ns_arr=("handover_block" "close_marker" "stack_blocks" "ziploc" "lift_ball" "rope" "pickup_plate" "stack_bowls" "insert_marker_into_cup" "insert_battery")

ns_arr=("stack_blocks" "close_marker")
for ns in "${ns_arr[@]}"
do
    for i in $(seq 1 25);
    do
        # python3 data_processing_bimanual_trajectory.py -t close_marker -d $i 
        # python3 data_processing_bimanual_keypose.py -t close_marker -d $i 
        # python3 data_processing_bimanual_keypose.py -t $ns -d $i 
        python3 data_processing_bimanual_trajectory.py -t $ns -d $i 

    done
done

# hard

# handover_block
# close_marker
# stack_blocks
# ziploc #need to customize EE
# insert_battery

# easy

# lift_ball #need to customize EE
# rope
# pickup_plate # might need to customize EE
# stack_bowls
# insert_marker_into_cup
