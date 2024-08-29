clear; clc;
syms q1 q2 q3 q4 q5 q6 real;
% inverse elbow

% q1 = 0.;
% q2 = 0.4;
% q3 = 0.;
% q4 = 0.5;
% q5 = 0.7;
% q6 = 0.0;

E7 =[
    ones(3,3),[0.1065 0.0 0]';
    [0 0 0 1] ];

%wrist_rotate
E6 =[
    ROTX(q6),[0.07 0.0 0]';
    [0 0 0 1] ];
%wrist_angle
E5 =[
    ROTY(q5),[0.1 0. 0]';
    [0 0 0 1] ];
%forearm_roll
E4 =[
    ROTX(q4),[0.2 0.0 0]';
    [0 0 0 1] ];
%elbow
E3 =[
    ROTY(q3),[0.0595 0.0 0.3]';
    [0 0 0 1] ];
% shoulder
E2 =[
    ROTY(q2),[0 0 0.04805]';
    [0 0 0 1] ];
% waist
E1 =[
    ROTZ(q1),[0. 0. 0.079]';
    [0 0 0 1] ];
ET = E1*E2*E3*E4*E5*E6*E7
ET = simplify(ET)

iET = inv(ET); 
JSdq1 = simplify(DESKEW4(diff(ET , q1)*iET));
JSdq2 = simplify(DESKEW4(diff(ET , q2)*iET));
JSdq3 = simplify(DESKEW4(diff(ET , q3)*iET));
JSdq4 = simplify(DESKEW4(diff(ET , q4)*iET));
JSdq5 = simplify(DESKEW4(diff(ET , q5)*iET));
JSdq6 = simplify(DESKEW4(diff(ET , q6)*iET));
JS = simplify([JSdq1,JSdq2,JSdq3,JSdq4,JSdq5,JSdq6])

JBdq1 = simplify(DESKEW4(iET*diff(ET , q1)));
JBdq2 = simplify(DESKEW4(iET*diff(ET , q2)));
JBdq3 = simplify(DESKEW4(iET*diff(ET , q3)));
JBdq4 = simplify(DESKEW4(iET*diff(ET , q4)));
JBdq5 = simplify(DESKEW4(iET*diff(ET , q5)));
JBdq6 = simplify(DESKEW4(iET*diff(ET , q6)));
JB = simplify([JBdq1,JBdq2,JBdq3,JBdq4,JBdq5,JBdq6])

