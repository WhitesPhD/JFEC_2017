% Kalman filter procedure to calcuate the PREDICTION, UPDATING and
% SMOOTHING equations and hence the mean bt|t and variance Pt|t of the state vector 
% and also draw the latent factor bt

% state - space is of the form
%    yt = Ht*bt + At*xt + et            et ~ N(0,R)
%    bt = mt + Ft*b(t-1)+ Gt*xt + ut    ut ~ Kt*N(0,Q)      
%    E(et,us) = 0  (but not necessarily, we could incorporate
%    correlation if we wanted to...)

% Prediction equations
%     bt|t-1 = mt + Ft*bt-1|t-1 + Gt*xt
%     Pt|t-1 = Ft*Pt-1|t-1*Ft' + diag(Kt)*Q
%     nt|t-1 = yt - yt|t-1 = yt - Ht*bt|t-1 - At*xt     PREDICTION ERROR
%     ft|t-1 = Ht*Pt|t-1*Ht' + R           CONDITIONAL VARIANCE
%     OF THE PREDICTION ERROR

% Updating equations
%     bt|t = bt|t-1 + Kt*nt|t-1
%     Pt|t = Pt|t-1 - Kt*Ht*Pt|t-1

% Kalman Gain
%     Kt = Pt|t-1*Ht'*(ft|t-1)^(-1)

% Smoothing equations (t = T-1, T-2, ... , 1)
%     bt|T = bt|t + Pt|t*Ft'*(P(t+1)|t)^(-1)*(b(t+1)|T - Ft*bt|t-mt)
%     Pt|T = Pt|t + Pt|t*Ft'*(P(t+1)|t)^(-1)*(P(t+1)|T -
%     P(t+1)|t)*(P(t+1)|t)^(-1)'*Ft*Pt|t'

function [bt, b_post, Pt, ft_tlag]  = FFBS(Yt, Ht, At, Xt, R, mt, Ft, Gt, K, Q, b0, P0, ndraws) 

% INPUTS

%      Yt: is the observed data matrix of size  T x n 
%      where    n: number of observed variables, T: number of observations
%      Xt: is the observed exogenous variables matrix of size T x m  
%      where    m: number of observed exogenous variables

%      b0: Initial value for state vector bt of size nstates x 1 

%      Yt:     T x n 
%      Ht:     T x n x nstates
%      At:     n x m 
%      Xt:     T x m
%      R:      n x n  
%      mt:     nstates x 1
%      Ft:     nstates x  nstates
%      Gt:     nstates x m 
%      K:      T x nstates
%      Q:      nstates x  nstates
%      b0:     nstates x 1  
%      P0:     nstates x nstates 

% OUTPUTS

%      bt:     unsmoothed Kalman beta estimates 
%      Pt:     unsmoothed Kalman covariance matrix estimates 
%      bt_T:   smoothed Kalman beta estimate 
%      Pt_T:   smoothed Kalman covariance matrix estimates 

T = size(Yt,1);
n = size(Yt,2);
nstates = size(mt,1);

% denote bt|t-1 as bt_tlag   T    x  nstates
% denote bt|t   as bt_t      T+1  x  nstates
% denote Pt|t-1 as Pt_tlag   T    x  nstates  x  nstates 
% denote Pt|t   as Pt_t      T+1  x  nstates  x  nstates
% denote nt|t-1 as nt_tlag   T    x  n
% denote ft|t-1 as ft_tlag   T    x  n   x  n
% Kalman gain   Kt           T    x  nstates  x  n

%Initialize state vector bt
bt_tlag = zeros(T,nstates);
bt_t = zeros(T+1, nstates);
bt_t(1,:) = b0';

%Initialize state vector variance Pt
Pt_tlag = zeros(T, nstates, nstates);
Pt_t = zeros(T+1, nstates, nstates);
Pt_t(1,:,:) = P0;

%Initialize prediction error nt
nt_tlag = zeros(T, n);

%Initialize variance of the prediction error ft 
ft_tlag = zeros(T, n, n);

%Initialize Kalman gain vector
Kt = zeros(T, nstates, n);

% *********************************Running the Kalman Filter
for i=1:T
    % Prediction equations    
    if size(Xt,1)> 0
        bt_tlag(i,:) = (mt + Ft*bt_t(i,:)' + Gt*Xt(i,:)')';   
    else
        bt_tlag(i,:) = (mt + Ft*bt_t(i,:)')'; 
    end

    Pt_tlag(i,:,:) = Ft*reshape(Pt_t(i,:,:), nstates, nstates)*Ft' + diag(K(i,:))*Q;
    if size(Xt,1) > 0
        nt_tlag(i,:) = (Yt(i,:)' - reshape(Ht(i,:,:),n,nstates)*bt_tlag(i,:)' - At*Xt(i,:)')';   
    else
        nt_tlag(i,:) = (Yt(i,:)' - reshape(Ht(i,:,:),n,nstates)*bt_tlag(i,:)')';    
    end
    
    %ft_tlag(i,:,:) = reshape(Ht(i,:,:),n,nstates)*reshape(Pt_tlag(i,:,:), nstates, nstates)*reshape(Ht(i,:,:),n,nstates)' + R;
    ft_tlag(i,:,:) = reshape(Ht(i,:,:),n,nstates)*reshape(Pt_tlag(i,:,:), nstates, nstates)*reshape(Ht(i,:,:),n,nstates)' + reshape(R(i,:,:),n,n);

    % Kalman gain
    Kt(i,:,:) = reshape(Pt_tlag(i,:,:), nstates, nstates)*reshape(Ht(i,:,:),n,nstates)'*inv(reshape(ft_tlag(i,:,:),n,n));

    % Updating equations
    bt_t(i+1, :) = (bt_tlag(i,:)' + reshape(Kt(i,:,:), nstates, n)*nt_tlag(i,:)')';  
    Pt_t(i+1,:,:) = ((reshape(Pt_tlag(i,:,:), nstates, nstates)- reshape(Kt(i,:,:),nstates,n)*reshape(Ht(i,:,:),n,nstates)*reshape(Pt_tlag(i,:,:),nstates,nstates)));
end    

bt = bt_t;
Pt = Pt_t;

% *******************************Running the Kalman Smoother

nobs = size(bt,1);
nstates = size(bt,2);

% get bT|T and PT|T from the last iteration of the kalman filter

bT_T = bt_t(end,:);
PT_T = squeeze(Pt_t(end,:,:));

% ***********************************************************
nobs = nobs-1;

% initialize array that will store the numerical posterior distribution:
% nobs x nstates       is one path of this distribtuion.....
% .... and we draw       ndraws    of these paths
bt_path = zeros(ndraws, nobs, nstates);

for i = 1:ndraws

             for ns=1:nstates
                    bt_path(i, nobs, ns) = normrnd(bT_T(1,ns), PT_T(ns,ns));
             end
    
    for j = 0:nobs-2
        part1 = reshape(Pt_t(nobs-1-j, :, :), nstates, nstates) *Ft';
        part2 = (Ft*reshape(Pt_t(nobs-1-j, :, :), nstates, nstates)*Ft'+diag(K(nobs-1-j,:))*Q)\eye(nstates);

        if size(Xt,1)>0
            part3 = reshape(bt_path(i, nobs-j,:),1,nstates)' - mt - Ft*bt_t(nobs-1-j,:)'-Gt*Xt(nobs-1-j,:)';       
        else
            part3 = reshape(bt_path(i, nobs-j,:),1,nstates)' - mt - Ft*bt_t(nobs-1-j,:)';   
        end
       
        bt_tpl1   = bt_t(nobs-1-j,:)' + part1*part2*part3;

        part4    =  reshape(Pt_t(nobs-1-j,:,:), nstates, nstates)*Ft';
        part5    =  part2;
        Pt_tpl1  =  reshape(Pt_t(nobs-1-j,:,:), nstates, nstates) - part4*part5*part4';
        
   
%         
        % now draw bt for t = T-1, ... , 1
        if(sum(K(nobs-1-j,:),2)==nstates)
        
        [~, h]  = chol(Pt_tpl1);
        
        if h ~= 0
            Pt_tpl1(isnan(Pt_tpl1)) = PT_T(isnan(Pt_tpl1));
            Pt_tpl1(isinf(Pt_tpl1)) = PT_T(isinf(Pt_tpl1));
            [V,D] = eig(Pt_tpl1);
            V1 = V(:,1);
            C2 = Pt_tpl1 + V1*V1'*(eps(D(1,1))-D(1,1));
            Pt_tpl1 = C2;
        end 
    
            bt_path(i, nobs-1-j, :) = mvnrnd(zeros(1,size(bt_tpl1,1)),eye(size(Pt_tpl1,1)))*chol(Pt_tpl1)+bt_tpl1';
        else
            for ns=1:nstates
                if(K(nobs-1-j,ns)==1)
                    bt_path(i, nobs-1-j, ns) = normrnd(bt_tpl1(ns,1), Pt_tpl1(ns,ns));
                else
                    bt_path(i, nobs-1-j, ns) = bt_tpl1(ns,1);
                end
            end
        end
    end
end
b_post = bt_path;
end
