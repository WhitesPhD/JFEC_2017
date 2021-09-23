% Gerlach - Carter - Kohn algorithm to simulate structural breaks
% 
% state - space is of the form:
% (1) y_t = g_t + h'_t*z_t + G_t*u_t
% (2) z_t = f_t + F_t*z_t-1 + Gam_t*u_t
% 
% where th_t= ( g_t, h_t, G_t, f_t, F_t, Gam_t) and t=1,...,T
%
%
% Backwards 3-steps precedure
%
% STEP 1:
% Fix Om_T=0 and nu_T =0, then for t=T-1,...,1 %
% Om_t = A'_t+1*( Om_t+1 - Om_t+1*C_t+1*D_t+1^-1*C'_t+1*Om_t+1 )*A_t+1 + F'_t+1*h_t+1*h'_t+1*F_t+1 / r_t+1 %  
% nu_t = A'_t+1*( I - Om_t+1*C_t+1*D_t+1^-1*C'_t )*( nu_t+1 - Om_t+1* ( a_t+1 + O_t+1*y_t+1 ) ) + F'_t+1*h_t+1* ( y_t+1 - g_t+1 - h'_t+1*f_t+1 ) / r_t+1 %
%
% where
% O_t+1 = Gam_t+1 * ( Gam'_t+1*h_t+1 + G'_t+1 ) / r_t+1
% a_t+1 = ( I - O_t+1*h'_t+1)*f_t+1 - O_t+1*g_t+1 %
% A_t+1 = (I - O_t+1*h'_t+1)*F_t+1 %
% C_t+1*C'_t+1 = var( z_t+1 | z_t, y_t+1, th_t+1 ,K) 
%              = Gam_t+1* ( I - ( 1 / r_t+1 )*( Gam'_t+1*H_t+1 + G'_t+1 )*( Gam'_t+1*H_t+1 + G'_t+1 )') *Gam'_t+1 %
% C_t+1 = chol( C_t+1*C'_t+1 )' %
% D_t+1 = I +  C'_t+1*Om_t+1*C_t+1 %
% r_t+1 = var( y_t+1 | z_t, ,K_t+1,1 )
%       = ( h'_t+1 * Gam_t+1 + G_t+1 )*( h'_t+1 * Gam_t+1 + G_t+1 )' %
%
% STEP 2:
% Given E(z_0) and var(z_0), and t=1,...,T %
% p( y_t | Y_t-1,1, th, K_t,1 ) ~ R_t^{1/2} * exp( -(1/(2*R_t))* ( y_t - g_t - h'_t*( f_t + F_t*m_t-1 ) )^2 ) %
% p( Y_t+1,T | Y_t,1, th, K ) ~ |T'_t*Om_t*T_t|^{-1/2} * exp( -(1/2)*(m'_t*Om_t*m_t - 2*nu'_t*m_t - ( nu_t - Om_t*m_t )'*T_t*(T'_t*Om_t*T_t)^{-1}*T'_t*( nu_t - Om_t*m_t ) %
%
% where
% m_t = E( z_t | Y_t,1, th, K ) 
%     = ( I - J_t*h'_t )*( f_t + F_t* m_t-1 ) + J_t*( y_t - g_t)%
% V_t = var( z_t | Y_t,1, th, K )
%     = F_t*V_t-1*F'_t + Gam'_t*Gam_t - J_t*J'_t*R_t %
% R_t = var( y_t | Y_t-1,1, th, K )
%     = h'_t*F_t*V_t-1*F'_t*h_t + ( h'_t*Gam_t + G_t )*( h'_t*Gam_t + G_t )' %
% J_t = ( F_t*V_t-1*F'_t*h_t + Gam_t*( h'_t*Gam_t + G_t )') / R_t %
% T = chol(V_t)' %
%
% STEP 3
% p( K_t | Y, th, K_s#t ) ~ p( Y_t+1,T | Y_t,1, th, K )* p( y_t | Y_t-1,1, th, K_t,1 )*p( K_t | th, K_s#t ) %
%
% where
% p( K_t | th, K_s#t ) is obtained from the prior

function Kt = breaks_sampler_aff_R(Y, g, h, z, G, f, F, Gam, z0, P0, K0, p, nsim);

% INPUTS

%      Y: is the observed data matrix of size  T x 1 
%      where    T: number of observations, nstates: number of states
%      z: the state vector of size T x nstates, z_t: nstates x 1
%      Gam: the parameter affected by Kt
%      z0: Initial value for state vector bt of size nstates x 1 
%      P0= var( z_0 | Y_0,1, th, K ) of size nstates x nstates
%      K0: Previous structural breaks T-1 x nstates
%      p: p(K_t=1 | th, Ks#t)
%
%      Y:     T x N, y_t: N x 1 
%      g:     T x N, g_t: N x 1
%      h:     T x N x nstates, h'_t: N x nstates 
%      G:     T x N x nstates, G_t: N x nstates  
%      f:     T x nstates, f_t= nstates x 1
%      F:     T x nstates x  nstates
%      Gam:   T x nstates x nstates (different in this particular example- see behind) 
%      K0:    T x nstates-N
%      Q:     nstates x  nstates
%      z0:    nstates x 1  
%      P0:    nstates x nstates 
%
% OUTPUTS
%
%      Kt:     vector of structural breaks T x nstates

[T,N] = size(Y);
nstates = size(z,2);
M=nstates-N+1;
% Initialize breaks vector Kt
Kt= zeros(nsim+1,T,M);
Kt(1,:,:)=K0;

for i_sim=2:nsim+1
    % STEP 1
    Om = zeros(T+1,nstates,nstates);
    nu = zeros(T+1,nstates);
    for j=1:T
        % reshape variables
        Gamt=Gam(T+1-j,:); 
        %Gamt=[Gam(1:N,1:N),zeros(N,M);zeros(M,N),diag(reshape(Kt(i_sim-1,T+1-j,:),1,M)).*Gam(N+1:nstates,N+1:nstates)];
        ht = reshape(h(T+1-j,:,:),N,nstates)';
        Gt = reshape(G(T+1-j,:,:),N,N);
        ft = f(T+1-j,:)';
        gt = g(T+1-j,:)';
        Ft = reshape(F(T+1-j,:,:),nstates,nstates);
        Omt=reshape(Om(T+2-j,:,:),nstates,nstates);
    
        % useful vectors-matrices
        rt = (ht'*Gamt+Gt)*(ht'*Gamt+Gt)';
        Ot = Gamt *(Gamt'*ht + Gt')*inv(rt);
        at = (eye(nstates) - Ot*ht')*ft - Ot*gt;
        At = (eye(nstates) - Ot*ht')*Ft;
        Ct2 = Gamt*( eye(nstates) - ( Gamt'*ht+ Gt' )*inv(rt)*( Gamt'*ht+ Gt' )') *Gamt';
        Ct = Ct2^0.5;
        Dt = eye(nstates) +  Ct'*Omt*Ct;
    
        % main results
        Om(T+1-j,:,:) = At'*( Omt - Omt*Ct*inv(Dt)*Ct'*Omt )*At + Ft'*ht*inv(rt)*ht'*Ft ;
        nu(T+1-j,:) = (At'*(eye(nstates)-Omt*Ct*inv(Dt)*Ct')*( nu(T+2-j,:)'-Omt*(at+Ot*Y(T+1-j,:)') )+Ft'*ht*inv(rt)*(Y(T+1-j,:)'-gt-ht'*ft))'; 
    end
    Om=Om(1:T,:,:);
    nu=nu(1:T,:);
    
    
    % STEP 2-3
    Vt = zeros(T+1,nstates,nstates);
    mt = zeros(T+1,nstates);
    Vt(1,:,:) = P0;
    mt(1,:) = z0';
    for i=2:T+1   % First t=1 to start Rt
        
        % 0 refers to the case K_t=0, 1 to K_t=1
        % reshape variables
        Omt=reshape(Om(i-1,:,:),1,1);
        nut=nu(i-1,:)';
        ht = reshape(h(i-1,:,:),N,nstates)';
        Gt = G(i-1);                           %%%% CORRECTED %%%%%
        ft = 0;                 % always 0 in this example
        gt = 0;                    % always 0 in this example
        Ft = reshape(F(i-1,:,:),nstates,nstates);
        Gamt0=0;
        Gamt1=Gam(i-1,:);
        mt_1 = mt(i-1,:)';
        % general to compute Rt and Jt
        htG = reshape(h(i-1,:,:),N,nstates)';
        FtG = reshape(F(i-1,:,:),nstates,nstates);
        GamtG0 = 0;
        GamtG1 = 1*Gam(i-1,:); %%%% CORRECTED %%%%%
        VtG_1 = reshape(Vt(i-1,:,:),nstates,nstates);

        Rt0 = htG'*(FtG*VtG_1*FtG')*htG + (htG'*GamtG0)*(htG'*GamtG0)'+Gt*Gt';  %%%% CORRECTED %%%%
        Jt0 = ((Ft*VtG_1*Ft')*ht+Gamt0'*Gamt0*ht)*inv(Rt0);%%%% CORRECTED %%%%
        V0 = Ft*VtG_1*Ft'+ Gamt0'*Gamt0-Jt0*Rt0*Jt0';
        m0 = ((eye(nstates)-Jt0*ht')*(ft+Ft*mt_1)+Jt0*(Y(i-1,:)'-gt));
        Tt0=V0^0.5;

        Rt1 = htG'*(FtG*VtG_1*FtG')*htG + (htG'*GamtG1)*(htG'*GamtG1)'+Gt*Gt';  %%%% CORRECTED %%%%
        Jt1 = ((Ft*VtG_1*Ft')*ht + Gamt1*(Gamt1'*ht+Gt'))*inv(Rt1);
        V1 = Ft*VtG_1*Ft'+ Gamt1'*Gamt1-Jt1*Rt1*Jt1';
        m1 = ((eye(nstates)-Jt1*ht')*(ft+Ft*mt_1)+Jt1*(Y(i-1,:)'-gt));
        Tt1 = V1^0.5;

        % main results STEP 2
        py0=(-1/2)*log(2*pi*Rt0)-0.5*(Y(i-1,:)'-gt-ht'*(ft+Ft*mt_1))'*inv(Rt0)*(0.5*(Y(i-1,:)'-gt-ht'*(ft+Ft*mt_1)));
        if(isnan(py0)==1)
            py0=0;
        end    
        pY0=-0.5*log((Tt0'*Omt*Tt0+eye(nstates)))-(1/2)*(m0'*Omt*m0-2*nut'*m0-(nut-Omt*m0)'*Tt0*inv(Tt0'*Omt*Tt0+eye(nstates))*Tt0'*(nut-Omt*m0));
        if(isnan(pY0)==1)
            pY0=0;
        end
        py1=(-1/2)*log(2*pi*Rt1)-0.5*(Y(i-1,:)'-gt-ht'*(ft+Ft*mt_1))'*inv(Rt1)*(Y(i-1,:)'-gt-ht'*(ft+Ft*mt_1));
        if(isnan(py1)==1)
            py1=0;
        end
        pY1=-0.5*log((Tt1'*Omt*Tt1+eye(nstates)))-(1/2)*(m1'*Omt*m1-2*nut'*m1-(nut-Omt*m1)'*Tt1*inv(Tt1'*Omt*Tt1+eye(nstates))*Tt1'*(nut-Omt*m1));
        if(isnan(pY1)==1)
            pY1=0;
        end
        
        % main results STEP 3
        pKt0=exp(py0-py1)*exp(pY0-pY1)*(1-p);
        pKt1=p;
        pKt=(pKt1/(pKt0+pKt1));
        u=unifrnd(0,1);    % uniform number
        if(pKt<u)
            Kt(i_sim,i-1,1)=0;
            Vt(i,N,N)=V0(N,N);
            mt(i,N)=m0(N,1);
        else
            Kt(i_sim,i-1,1)=1;
            Vt(i,N,N)=V1(N,N);
            mt(i,N)=m1(N,1);
        end
            
    end
end
% discard starting value
Kt=Kt(2:nsim+1,:,:);
    
    
    






