for i in $(seq 1 50);
do
    #python3 data_processing_bimanual_keypose.py -t hand_over_block -d $i

    #python3 data_processing_bimanual_keypose.py -t stack_bowl -d $i
    #python3 data_processing_bimanual.py -t stack_bowl_single_arm -d $i
    
    #python3 data_processing_bimanual_keypose.py -t single_arm -d $i
    #python3 data_processing_bimanual.py -t single_arm -d $i

    #python3 data_processing_bimanual.py -t close_pen -d $i
    # python3 data_processing_bimanual_keypose.py -t close_pen -d $i

    #python3 data_processing_bimanual.py -t pick_up_plate -d $i
    python3 data_processing_bimanual_keypose.py -t pick_up_plate -d $i
    
    #python3 data_processing_bimanual.py -t pouring_into_bowl -d $i
    # python3 data_processing_bimanual_keypose.py -t pouring_into_bowl -d $i

    #python3 data_processing_bimanual.py -t put_block_into_bowl -d $i
    # python3 data_processing_bimanual_keypose.py -t put_block_into_bowl -d $i

    #python3 data_processing_bimanual.py -t stack_block -d $i
    #python3 data_processing_bimanual_keypose.py -t stack_block -d $i
done

#    python3 data_processing_bimanual.py -t stack_block -d $i

