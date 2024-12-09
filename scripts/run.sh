for i in $(seq 1 45);
# for i in $(seq 1 2);
do
    python3 data_processing_bimanual_keypose.py -t pour_into_bowl -d $i 
    # python3 data_processing_bimanual_keypose.py -t pick_up_notebook -d $i 
    # python3 data_processing_bimanual_keypose.py -t open_marker -d $i 
    # python3 data_processing_bimanual_keypose.py -t open_pill_case -d $i 
    # python3 data_processing_bimanual_keypose.py -t straighten_yellow_rope -d $i 
    #python3 data_processing_bimanual_keypose.py -t lift_ball -d $i
    #python3 data_processing_bimanual_keypose.py -t stack_blocks -d $i
    #python3 data_processing_bimanual_keypose.py -t close_marker -d $i
    #python3 data_processing_bimanual_keypose.py -t stack_bowls -d $i
    
    #python3 data_processing_bimanual_keypose.py -t hand_over_block -d $i
    #python3 data_processing_bimanual_keypose.py -t pick_up_plate -d $i
done



