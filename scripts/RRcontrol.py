function finalerr = RRcontrol(gdesired, K)
    dist_threshold = 0.005;  % Define distance threshold
    angle_threshold = (0.15*pi)/180;% Define angle threshold
    Tstep = 0.1;    % Define T_step
    %R_position = tf_frame('base_link', 'R', gdesired);
    maxiter = 1000;

    q = [0.3190680146217346, -0.43411657214164734, 0.8789710402488708, 0.647339940071106, 0.26077672839164734, -0.5737088322639465]

    %  Foor loop will run till the convergence or max iteration
    for iteration = 1:maxiter
        gst = FwdKin(q);
        err = inv(gdesired) * gst;
        xi = getXi(err);
        
        J = BodyJacobian(q) ;

        q = double(q - K * Tstep * pinv(J)*xi);
        
        J = BodyJacobian(q) ;
        if abs(det(J))<0.01
            fprintf('Singularity position \n');
            finalerr = -1;
            return;
        end
        
        if norm(xi(1:3)) < dist_threshold && norm(xi(4:6)) < angle_threshold
            finalerr = norm(xi(1:3))*10;  
            fprintf('Convergence achieved. Final error: %.2f cm\n', finalerr);
            return;
        end
        finalerr = norm(xi(1:3))*10;
        fprintf('finalerr: %.3f\n', finalerr);
        %ur5.move_joints(q,0.5);

        %pause(0.5);
    end

    finalerr = -1;
end


