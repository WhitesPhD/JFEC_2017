%% ReadMe

% The codes in this folder estimate the benchmark model specification with
% Bayesian Model Averaging with Stochastick Break Betas and Stochastic
% Break Volatilities (BMA-SBB-SBV) as in Bianchi, Guidolin, Ravazzolo 2017 
% "Dissecting the 2007-2009 Real Estate Market Bust: Systematic Pricing 
% Correction or Just a Housing Fad?", Journal of Financial Econometrics 
% Please cite the paper if you use the code

% The file REIT.xlsx contains data on sector tax-qualified REIT total returns 
% obtained from the North American Real Estate Investment Trust (NAREIT) Association 
% and consist of data on 11 portfolios formed when REITs are classified on the basis 
% of their main focus of activity, i.e., Industrial, Office, Shopping Centers, Regional
% Malls, Free Standing shops, Apartments, Manufactured Homes, Healthcare, Lodging/Resorts, 
% Self-Storage and Mortgage REITs. Apartments and Manufactured Homes represent 
% the “Residential” real estate sector. Mortgage REITs specialize in mortgage-backed security investments.
% Data on Equity portfolios can be found online on the Ken French website.
% We provide a complete description of the data on macroeconomic risk
% factors in the paper. 

% The Matlab codes are organised as follows:

% GibbsSampler.m
% this is the main code to run, it requires as input the data file and 
% performs the estimation according to the Gibbs Sampler outlined in 
% the Appendix of the paper. At the end of the MCMC loop, the code saves 
% the MCMC draws after the burnin and for a given thinning step for the 
% conditional pricing errors, betas and idiosyncratic risks. We exploit the
% triangularization scheme proposed by Carriero, Clark and Marcellino 2015 
% which allows to estimate the model equation by equation.

% The main code calls the following subroutines:
% 1. break_sampler_aff.m
% this function represents the Gerlach - Carter - Kohn algorithm to simulate 
% structural breaks for the conditional betas.

% 2. FFBS.m
% this routine performs a draw for the latent states conditional on the
% parameters, the latent stochastic breaks and the conditional
% idiosyncratic volatility.

% 3. break_sampler_aff_R.m
% this function represents a modified version of the Gerlach - Carter - Kohn 
% algorithm to simulate structural breaks for the idiosyncratic risks. 

% 4. randgamma.m
% function to simulate from a Gamma distribution

% 5. mixtures.m
% this routine performs a draw for the mixture components of the volatilities 
% using the algorithm proposed by Omori, Chib, Shepard, and Nakajima 2007,

% Finally note that some parts of this code might require the Matlab
% statistical toolbox (but only for the graphs, not the estimation part). 
% On a fairly new laptop estimation should not take long.


