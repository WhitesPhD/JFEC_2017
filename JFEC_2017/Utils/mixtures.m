% Purpose: Sampling from a mixture of 10 normal distribution to
% approximate a log chi2
%--------------------------------------------------------------------------
% Usage: [res k]= mixtures(lnsig,Y,ind)
%--------------------------------------------------------------------------
% Input: 
%--------------------------------------------------------------------------
% lnsig = [T x 1] vector of log volatilities
% Y     = [T x 1] vector of observations
% ind   = flag to identify the sampling options, 'O' Omori et al. 2007
%--------------------------------------------------------------------------
% Output:
% res   = [T x 1] vector of selected means and stds of the mixture
% k     = [T x 1] vector of drawing from the multinomial
%--------------------------------------------------------------------------
% References:
% Omori, Chib, Shephard, Nakajima (2007), Journal of Econometrics
% " Stochastic volatility with leverage: Fast and efficient likelihood
% inference"
%--------------------------------------------------------------------------

function [res, k]= mixtures(lnsig,Y,ind)
 
  [T] = size(Y,1);
  
  if strcmp(ind,'O')
   mu        = [1.92677 1.34744 0.73504 0.02266 -0.85173 -1.97278 -3.46788 -5.55246 -8.68384 -14.65];
   ss2       = [0.11265 0.17788 0.26768 0.40611 0.62699 0.98583 1.57469 2.54498 4.16591 7.33342];
   q         = [0.00609 0.04775 0.13057 0.20674 0.22715 0.18842 0.12047 0.05591 0.01575 0.00115];
  else
   mu        = [1.5074 0.5247 -0.6509 -2.3585 -4.2432 -9.8372 -11.4004];
   ss2       = [0.1673 0.3402 0.6400 1.2626 2.6137 5.1795 5.7959];
   q         = [0.04395 0.24566 0.34001 0.2575 0.10556 0.00002 0.0073];
  end
  
  sig       = sqrt(ss2);
  
  temp      = Y - lnsig;
  w         = zeros(T,size(q,2));
  k         = zeros(T,1);
  res       = zeros(T,2);
    
  for i=1:T
  w(i,:)      = normpdf(temp(i),mu,sig).*q;
  if ~(sum(w(i,:)) > 0) || ~all(w(i,:)>=0) % catches missing values
  w(i,:)      = ones(1,size(q,2))*1/size(q,2);
  end
  k(i)        = randsample(size(q,2),1,true,w(i,:));
  res(i,:)    = [mu(k(i)) ss2(k(i))]; 
  end
   
end  
  
  