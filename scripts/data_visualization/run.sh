for i in $(seq 1 1);
do
    python3 data_visualization_mosaic.py -t close_marker -d $i
    python3 data_visualization_mosaic.py -t hand_over_block -d $i
    python3 data_visualization_mosaic.py -t lift_ball -d $i
    
    python3 data_visualization_mosaic.py -t open_marker -d $i
    python3 data_visualization_mosaic.py -t open_pill_case -d $i
    python3 data_visualization_mosaic.py -t pick_up_notebook -d $i

    python3 data_visualization_mosaic.py -t pick_up_plate -d $i
    python3 data_visualization_mosaic.py -t pour_into_bowl -d $i
    python3 data_visualization_mosaic.py -t stack_blocks -d $i

    python3 data_visualization_mosaic.py -t stack_bowls -d $i
    python3 data_visualization_mosaic.py -t straighten_rope -d $i  
done
