% ************************************************************************************
% Dynamic factor model with Bayesian model averaging and stochastic breaks in the 
% conditional betas and volatilities. 
%
% Bianchi, Guidolin, Ravazzolo 2017 "Dissecting the 2007-2009 Real Estate Market Bust: Systematic
% Pricing Correction or Just a Housing Fad?". Please cite the paper if you use the code or part of it.
% 
% This code performs the Gibbs sampler for the estimation of the most
% general model specification. The triangularization scheme follows
% Carriero, Clark and Marcellino 2015 and allows to estimate the model
% equation by equation.
% 
% For bugs and requests:
% Daniele Bianchi, University of Warwick, daniele.bianchi@wbs.ac.uk
% ************************************************************************************

close all; clear all; clc; randn('seed',3123), rand('seed',3123), warning off

%--------------------------------------------------------------------------
% Load data
%--------------------------------------------------------------------------

% load data.mat
 
% X = RiskFactors;
% Y = Portfolios;

X      = [ones(size(X,1),1) X];     % Matrix of covariates

% Calibration sample (to calibrate priors)

T0           = 60;
Yprior       = Y(1:T0,:);
Xprior       = X(1:T0,:);

% Estimation sample (to estimate the model)

Yuse       = Y(T0+1:end,:);
Xuse       = X(T0+1:end,:);

[T, N]     = size(Yuse);
[~, K]     = size(Xuse);           % K = number of factors + intercept

%--------------------------------------------------------------------------
% Set MCMC 
%--------------------------------------------------------------------------

Ndraws      =   50000;      % # MCMC draws 
Burnin      =   10000;      % the burn-in period MCMC sampler
b_draws     =   10;        % thin 
b_seldr     =   [Burnin+b_draws:b_draws:Ndraws]';  % Final draws

%--------------------------------------------------------------------------
% Room saving
%--------------------------------------------------------------------------
    
MCMC_R              =  zeros(T,N,Ndraws);         % Returns volatilities
MCMC_lambda         =  zeros(Ndraws,T,N,2);       % Component of the chi2 approximation
MCMC_Q              =  zeros(Ndraws,N,K+1);       % State equations volatilities
MCMC_B              =  zeros(Ndraws,T,N,K);     % Draws of the betas from the posterior
MCMC_K              =  zeros(Ndraws,T,N,K+1);     % Draws of the betas from the posterior
MCMC_prob           =  zeros(Ndraws,N,K+1);       % Break probs
MCMC_i              =  ones(Ndraws,N,K);        % Model indicator
Resid_1             =  zeros(T,N);                % reduced form residuals

%--------------------------------------------------------------------------
% Set the prior hyper-parameters
%--------------------------------------------------------------------------

% Beta parameters for p(K_t=1) in the conditional betas

p0a         =   5;     
p0b         =   80;     

% Beta parameters for p(K_t=1) in the conditional volatilities

p0sva       =   5;     
p0svb       =   70;     

% Prior degrees of freedom for the conditional variances of pricing errors,
% betas and time-varying volatilities

df          = 100;
dfa         = 100;
dfv         = 100;

%--------------------------------------------------------------------------
% Pre-sample prior calibration
%--------------------------------------------------------------------------

temp         = eye(K)/(Xprior'*Xprior);                 
bOLS         = temp*Xprior'*Yprior;
resid        = Yprior - Xprior*bOLS;
s2OLS        = diag((resid'*resid)/(T0 - K - 1));

VbOLS        = zeros(K,K,N);
Qprior       = zeros(N,K);

for i=1:N

VbOLS(:,:,i) = s2OLS(i)*temp;
Qprior(i,:)  = diag(VbOLS(:,:,i))';

if i>1
    MU_A(1:i-1,i) = zeros(i-1,1);            %#ok<SAGROW> % prior mean of A 
    OMEGA_A_inv(1:i-1,1:i-1,i) = 0.001*eye(i-1); %#ok<SAGROW> % prior precision of A 
end

end

A_                  = eye(N);

%--------------------------------------------------------------------------
% Drawing the starting values from the priors
%--------------------------------------------------------------------------        

a  = 0;
aR = 0;

pki                   =  betarnd(p0a,p0b,N,K);  
pkiR                  =  betarnd(p0sva,p0svb,1,N); 

for ii = 1:N
        
        MCMC_B(1,:,ii,:)       =  ones(T,1)*reshape(zeros(1,K),1,K);
        MCMC_lambda(1,:,ii,:)  =  3*ones(T,2);                           
        MCMC_Q(1,ii,:)         =  [Qprior(ii,:) mean(s2OLS(ii,:))];     % Initial value for variances
        MCMC_R(:,ii,1)         =  ones(T,1)*s2OLS(ii,:);                % Initial value for returns volatilities
        MCMC_i(1,ii,:)         =  ones(1,K);
        MCMC_K(i,:,ii,1:K)     =  binornd(1,ones(T,1)*pki(ii,:),T,K); % First drawing of the breaks for pricing errors and betas
        MCMC_K(i,:,ii,K+1)     =  binornd(1,ones(T,1)*pkiR(ii),T,1);    % First drawing of the breaks for conditional variances
        
if ii > 1
    
        alphadraw   = MU_A(1:ii-1,ii)+chol(OMEGA_A_inv(1:ii-1,1:ii-1,ii),'lower')*randn(ii-1,1);
        A_(ii,1:ii-1)= -1*alphadraw';
    
end

end

invA_ = A_\eye(N);
pis   =[1,0.5*ones(1,K-1)];



%%
for i = 1:Ndraws-1

%----------------------------------------------------------------------------------------------
% Given the triangularization step we can estimate the model eq-by-eq
%----------------------------------------------------------------------------------------------


    for ii = 1:N 

    clc;
    display(['Percentage of the loop computed: ', num2str([i/Ndraws]*100)])
    display(['Asset number: ', num2str(ii) , ' checking breaks %: ', num2str([a./T aR./T])])
    display(['Asset number: ', num2str(ii) , ' factors selected %: ', num2str(find(MCMC_i(i,ii,:)==1)')])
    

% -------------------------------------------------------------------------                     
% Triangularization step as in Clark, Carriero and Marcellino 2017
% -------------------------------------------------------------------------

    Yuse_adj  = Yuse(:,ii);  
    if ii==1; Resid = zeros(T,N);
    else
        
        for l=1:ii-1
            Yuse_adj = Yuse_adj - invA_(ii,l)*(sqrt(MCMC_R(:,l,i)).*Resid(:,l));
        end
    end
    Yuse_adj = Yuse_adj./sqrt(MCMC_R(:,ii,i)); 
    Xuse_adj = Xuse./repmat(sqrt(MCMC_R(:,ii,i)),1,size(Xuse,2));
  
% -------------------------------------------------------------------------                     
% Model Selection step 
% -------------------------------------------------------------------------
    
            gamma_draw = squeeze(MCMC_i(i,ii,:));
            iS=int8(random('Uniform',1.5001,K+0.4999,1,1));
            iPrt1 = gamma_draw;
            iPrt1(iS)=1;
            isumi1   = zeros(T,1);
            ilati1   = zeros(T,1);
            isumi1(1,:) = Yuse(1,ii)-Xuse(1,:)*(iPrt1.*squeeze(MCMC_B(i,1,ii,:)));
            ilati1(1,:) = (-0.5*(iPrt1.*squeeze(MCMC_B(i,1,ii,:))-iPrt1.*squeeze(MCMC_B(1,1,ii,:)))'*(diag(squeeze(MCMC_K(i,1,ii,1:K))).*inv(diag(squeeze(MCMC_Q(i,ii,1:K)))))*(iPrt1.*squeeze(MCMC_B(i,1,ii,:))-iPrt1.*squeeze(MCMC_B(1,1,ii,:))));
     
            iPrt0       = gamma_draw;
            iPrt0(iS)   = 0;
            
            isumi0      = zeros(T,1);
            ilati0      = zeros(T,1);
            isumi0(1,:) = Yuse(1,ii)-Xuse(1,:)*(iPrt0.*squeeze(MCMC_B(i,1,ii,:)));
            ilati0(1,:) = (-0.5*(iPrt0.*squeeze(MCMC_B(i,1,ii,:))-iPrt0.*squeeze(MCMC_B(1,1,ii,:)))'*(diag(squeeze(MCMC_K(i,1,ii,1:K))).*inv(diag(squeeze(MCMC_Q(i,ii,1:K)))))*(iPrt0.*squeeze(MCMC_B(i,1,ii,:))-iPrt0.*squeeze(MCMC_B(1,1,ii,:))));
      
            for gs=2:T;
                isumi1(gs,:)=Yuse(gs,ii)-Xuse(gs,:)*(iPrt1.*squeeze(MCMC_B(i,gs,ii,:)));
                isumi0(gs,:)=Yuse(gs,ii)-Xuse(gs,:)*(iPrt0.*squeeze(MCMC_B(i,gs,ii,:)));

                ilati1(gs,:)=(-0.5*(iPrt1.*squeeze(MCMC_B(i,gs,ii,:))-iPrt1.*squeeze(MCMC_B(i,gs-1,ii,:)))'*(diag(squeeze(MCMC_K(i,gs,ii,1:K))).*inv(diag(squeeze(MCMC_Q(i,ii,1:K)))))*(iPrt1.*squeeze(MCMC_B(i,gs,ii,:))-iPrt1.*squeeze(MCMC_B(i,gs-1,ii,:))));
                ilati0(gs,:)=(-0.5*(iPrt0.*squeeze(MCMC_B(i,gs,ii,:))-iPrt0.*squeeze(MCMC_B(i,gs-1,ii,:)))'*(diag(squeeze(MCMC_K(i,gs,ii,1:K))).*inv(diag(squeeze(MCMC_Q(i,ii,1:K)))))*(iPrt0.*squeeze(MCMC_B(i,gs,ii,:))-iPrt0.*squeeze(MCMC_B(i,gs-1,ii,:))));    
            end;
            
            isum1=sum(ilati1,1)-0.5*isumi1'*(isumi1./squeeze(MCMC_R(:,ii,i)));  %N=1, applied log transformation
            isum0=sum(ilati0,1)-0.5*isumi0'*(isumi0./squeeze(MCMC_R(:,ii,i)));  %N=1, applied log transformation
            aj=pis(1,iS);%*isum1;
            bj=(1-pis(1,iS))*(exp(isum0-isum1));
            p_j_tilde=aj/(aj+bj);
        
            gamma_draw(iS) = binornd(1,p_j_tilde);
            MCMC_i(i+1,ii,:)   = gamma_draw;
        
            ind_i    = squeeze(MCMC_i(i+1,ii,:));
            index_i  = find(MCMC_i(i+1,ii,:)==1);
            Kk = size(index_i,1);

% -------------------------------------------------------------------------                     
% Drawing the breaks as in Giordani and Kohn 2008
% -------------------------------------------------------------------------
    
    Qs        = squeeze(MCMC_Q(i,ii,1:K));                      % Select the state volatility for the ith MCMC draws
    Ks        = squeeze(MCMC_K(i,:,ii,1:K));                    % Select the values for the ith MCMC draws for the breaks of the factors
    bs        = squeeze(MCMC_B(i,:,ii,:));                      % Select the values for the ith MCMC draws for the betas
    Fbsiconst = [zeros(1,1+Kk);zeros(Kk,1),eye(Kk)];             
    Fbsi      = permute(repmat(Fbsiconst,[1,1,T]),[3,1,2]);        
    
    hbsi      = zeros(T,1,K+1);
    Y_stars   = zeros(T,1);
    Gams      = zeros(T,Kk+1,Kk+1);
        
        for qs=1:T
            temp = Xuse_adj(qs,:).*ind_i';
            hbsi(qs,1,:)  = [1,temp]; 
            Y_stars(qs,:) = Yuse_adj(qs)-temp*bs(qs,:)';    % Fitted value Yhat = XB
            Gams(qs,:,:)  = [reshape(MCMC_R(qs,ii,i),1,1)^0.5,zeros(1,Kk);zeros(Kk,1),diag(Qs(index_i).^(1/2))];
        end;  

     Z      = [Y_stars,bs(:,index_i)];
     nbs    = 1;                    
%         
     K_post   = breaks_sampler_aff(Yuse_adj,zeros(T,1),hbsi(:,:,[1; index_i]),Z,0,zeros(T,size(Z,2)),Fbsi,Gams,[0;bs(1,index_i)'],[s2OLS(ii),zeros(1,Kk);zeros(Kk,1),diag(Qs(index_i))],Ks(:,index_i),pki(ii,index_i)',nbs);
     K_post   = squeeze(K_post(nbs,:,:)); 

     K_post_1 =  binornd(1,ones(T,1)*pki(ii,:),T,K);       
     K_post_1(:,index_i) = K_post;

     MCMC_K(i+1,:,ii,1:K)  =  K_post_1;                             
    
% -------------------------------------------------------------------------                     
% Updating the posterior breaks probabilities
% -------------------------------------------------------------------------
        
        a                 = sum(K_post_1,1);
        pki               = betarnd(p0a,p0b,N,K);
        pki(ii,index_i)   = betarnd(p0a+a(index_i),p0b+T-a(index_i));                 
        MCMC_prob(i+1,ii,1:K) = pki(ii,:);
    
    
 % -------------------------------------------------------------------------
% Drawing the betas conditional on the returns volatilities, the breaks, etc..
% -------------------------------------------------------------------------

        Hkf                 =  hbsi(:,:,2:1+K).*repmat(MCMC_i(i+1,ii,:),size(hbsi,1),1);             

        b0 = bOLS(:,ii);
        P0 = squeeze(VbOLS(:,:,ii));
        
        [~, b_post ,~ ,~]   =  FFBS(Yuse_adj, Hkf, [], [], ones(T,1), zeros(K,1), eye(K), [], K_post_1,diag(Qs), b0, P0, 1); 
        b_post              = squeeze(b_post);        
        MCMC_B(i+1,:,ii,:)  = b_post;                               
         
% -------------------------------------------------------------------------
% Drawing the variance of the betas given the rest
% -------------------------------------------------------------------------
           
        bb                  = [b_post];
        Y_star              = diff(bb,1,1);  
        sQ                  = Y_star'*Y_star;    

        par1                = 0.5*(size(Y_star,1) + df);
        par1a               = 0.5*(size(Y_star,1) + dfa);

        par2                = 0.5*(Qprior(ii,2:K)*df + diag(sQ(2:end,2:end))'); 
        par2a               = 0.5*(Qprior(ii,1)*dfa + sQ(1,1)); 
        
        Qif_inv             = zeros(1,K);
        Qif                 = zeros(1,K);
        
        Qif_inv(1)          =     1/par2a * randgamma(par1a,1); %random draw from a gamma (a_post, 1/b_post)
        Qif(1)              =     1/Qif_inv(1);

        for j=1:K-1
        Qif_inv(j+1)        =     1/par2(j) * randgamma(par1,1); %random draw from a gamma (a_post, 1/b_post)
        Qif(j+1)            =     1/Qif_inv(j+1);
        end
        
        MCMC_Q(i+1,ii,1:K)   =   Qif;

        
        Resid(:,ii)   = Yuse_adj   - sum(Xuse_adj.*(squeeze(MCMC_B(i+1,:,ii,:)).*repmat(ind_i',size(MCMC_B,2),1)),2);
        Resid_1(:,ii) = Yuse(:,ii) - sum(Xuse.*(squeeze(MCMC_B(i+1,:,ii,:)).*repmat(ind_i',size(MCMC_B,2),1)),2);
        
        


    end
         
% -------------------------------------------------------------------------
% Recover the reduced form residuals
% -------------------------------------------------------------------------
    
         Resid_2 = Resid_1*A_';
         
         for ii=1:N
        
         Qs                  = MCMC_Q(i,ii,K+1);                % given the volatility for the sigmat state
         Ks                  = reshape(MCMC_K(i,:,ii,K+1),T,1); % given the state for the volatility dynamics
         Sigma2              = reshape(MCMC_R(:,ii,i),T,1);     % given the returns volatilities 
         Fbsi                = ones(T,1,1);
         hbsi                = ones(T,1);
         
         Y_sts1              = zeros(T,1);                 
         Y_stscal            = Resid_2(:,ii).^2;
         Y_sts1(Y_stscal==0) = 0;
         Y_sts1(Y_stscal~=0) = log(Y_stscal);

         Gams                = ones(T,1)*Qs.^(1/2);

        Y_sts                = (Y_sts1-squeeze(MCMC_lambda(i,:,ii,1))');%./sqrt(MCMC_lambda(i,:,ii,2)'); 

        Z                    = log(Sigma2);
        nbs                  = 1;
        
        K_post                  = breaks_sampler_aff_R(Y_sts,zeros(T,1),hbsi,Z,MCMC_lambda(i,:,ii,2)'.^0.5,zeros(T,size(Z,2)),Fbsi,Gams,0,Qs,Ks,pkiR(1,ii),nbs);%log(s2OLS)
        MCMC_K(i+1,:,ii,K+2)    = K_post';                       % setting the drawn state for the volatility

% -------------------------------------------------------------------------                     
% Updating the posterior breaks probabilities
% -------------------------------------------------------------------------
                
        aR                    = sum(K_post);
        pkiR(1,ii)            = betarnd(p0sva+aR,p0svb+T-aR);
        MCMC_prob(i+1,ii,K+1) = pkiR(1,ii);
    
% -------------------------------------------------------------------------
% Drawing the conditional volatilities given the rest
% -------------------------------------------------------------------------             
                                                   
        [~, LnSigma2_post ,~ ,~]  = FFBS(Y_sts, ones(T,1), [], [], MCMC_lambda(i,:,ii,2)', 0, 1, [], K_post', Qs, 0, 10, 1); 

        MCMC_R(:,ii,i+1)          = exp(LnSigma2_post');
        
% -------------------------------------------------------------------------
% Selecting the component of the mixture for the log volatility
% -------------------------------------------------------------------------
        
        [res , ~]                 = mixtures(LnSigma2_post',Y_sts1,'O');
        
        MCMC_lambda(i+1,:,ii,1)   = res(:,1);
        MCMC_lambda(i+1,:,ii,2)   = res(:,2);
        
% -------------------------------------------------------------------------
% Drawing the variance of the conditional volatilities
% -------------------------------------------------------------------------

        ll                   = LnSigma2_post';
        Y_star               = diff(ll);
        sQ                   = Y_star'*Y_star;
                
        par1R                = 0.5*(size(Y_star,1)+dfv);
        par2R                = 0.5*(sQ);

        
        QifR_inv  = 1/par2R * randgamma(par1R,1); 
        QifR = 1/QifR_inv;
        
        MCMC_Q(i+1,ii,K+1)      = QifR;

        
        end

% -------------------------------------------------------------------------
% Drawing the matrix A^-1
% -------------------------------------------------------------------------

        for ii = 2:N 

        Y_spread_adj     = Resid_1(:,ii)./sqrt(squeeze(MCMC_R(:,ii,i+1)));
        X_spread_adj     = []; 
        
        for vv = 1:ii-1  
            X_spread_adj = [X_spread_adj Resid_1(:,vv)./sqrt(squeeze(MCMC_R(:,ii,i+1)))]; 
        end
        
        ZZ = X_spread_adj'*X_spread_adj; Zz = X_spread_adj'*Y_spread_adj;

        Valpha_post     = (ZZ + OMEGA_A_inv(1:ii-1,1:ii-1))\eye(ii-1);
        alpha_post      = Valpha_post*(Zz + OMEGA_A_inv(1:ii-1,1:ii-1,ii)*MU_A(1:ii-1,ii));
    
        alphadraw       =  alpha_post+chol(Valpha_post,'lower')*randn(ii-1,1);
        A_(ii,1:ii-1)   = -1*alphadraw' ;
        
        end 
        invA_ = A_\eye(N);

end


% -------------------------------------------------------------------------
% Save the results
% -------------------------------------------------------------------------

MCMC_B     = MCMC_B(b_seldr,:,:,:);
MCMC_R     = MCMC_R(:,:,b_seldr);
MCMC_Q     = MCMC_Q(b_seldr,:,:);
MCMC_prob  = MCMC_prob(b_seldr,:,:);
MCMC_i     = MCMC_i(b_seldr,:,:);










