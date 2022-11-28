#: Last update 28 november 2022
#: Author: Nicolas Bianco (nicolas.bianco@phd.unipd.it)
#: [for any issue, feel free to mail me]

#: Example code for:
#: Variational Bayes for multivariate predictive regressions 
#: with continuous shrinkage priors (Bernardi and Bianchi and Bianco, 2022)

#: Required packages: 
#: Rcpp, RcppArmadillo, RcppEigen, RcppNumerical, BH, numDeriv, GIGrvg
#: ggplot2, reshape2, ggnewscale, mvnfast, Matrix, TeachingDemos
#: Download dependencies
list_of_packages = c("Rcpp", "RcppArmadillo", "RcppEigen", "RcppNumerical", "BH", 
                     "numDeriv", "GIGrvg", "ggplot2", "reshape2", "ggnewscale", 
                     "mvnfast", "Matrix", "TeachingDemos")
new_packages = list_of_packages[!(list_of_packages %in% installed.packages()[,"Package"])]
if (length(new_packages)>0) install.packages(new_packages)

#: Install and load package
install.packages("VBmlr_1.0.tar.gz",repos=NULL,source=TRUE)
library(VBmlr)

#: Get example quantities and
#: simulate some data from multivariate regression
#: with autoregressive components and covariates
Omega = Omega_example()
Phi = Phi_example()
Gamma = Gamma_example()

n = 500
d = nrow(Omega)
p = ncol(Gamma)

params = list(mu=rep(0,d),Phi=Phi,Gamma=Gamma,Omega=Omega)
data = ran_mlr(n,params)
Y = data$Y
X = data$X

#: Set hyperparameters
hyper = list(a_nu=0.1,b_nu=0.1,tau=10,ups=10,
             e1=0.001,e2=0.001,e3=1,
             h1=0.001,h2=0.001,h3=1)

#: Run the model with Horseshoe prior 
#: "normal" = gaussian prior
#: "lasso" = Bayesian lasso prior (Park and Casella, 2008; Leng, 2014)
#: "ng" = normal-gamma prior (Griffin and Brown, 2010)
#: "hs" = horseshoe prior (Carvalho et al, 2010)
system.time(mod <- VBmlr(Y,X,          #: The response Y, the covariates X
                         AR=TRUE,      #: Tell to the algo you want autoregressive effects
                         hyper=hyper,  #: Give hyperparameters
                         prior="hs",   #: Set the prior
                         #: Set convergence parameters/options
                         Tol_ELBO=0.01,Tol_Par=0.01,Trace=1))

#: See the increase in Lower Bound
plot(mod$lowerBoundIter,type='b',xlab='iteration',ylab='ELBO')

#: Compare true parameters with estimates
matrixplot(Omega,mod$Omega_hat)
matrixplot(Phi,mod$Mu_q_theta[,1:d])
matrixplot(Gamma,mod$Mu_q_theta[,(d+2):ncol(mod$Mu_q_theta)])

#: Achieve sparsity with SAVS algorithm and plot
#: true parameters with sparse estimates
Omega_sparse = t(diag(1,d)-SAVS(mod$Mu_q_beta,Y[,-1]))%*%diag(as.vector(mod$mu_q_nu))%*%(diag(1,d)-SAVS(mod$Mu_q_beta,Y[,-1]))
matrixplot(Omega,Omega_sparse)
matrixplot(Phi,SAVS(mod$Mu_q_theta[,1:d],Y[,-n]))
matrixplot(Gamma,SAVS(mod$Mu_q_theta[,(d+2):ncol(mod$Mu_q_theta)],X[,-n]))

#: Plot the approximating densities for the model parameters
#: beta_{j,k}, nu_{j,j} and theta_{j,k}
#: The function also provides mean, standard deviation and HPD(0.95)
qDensity(mod,"beta",c(2,1))
qDensity(mod,"nu",c(1,1))
qDensity(mod,"theta",c(1,1))

#: Get the Wishart approximation (with EP) to the precision matrix
qOm = qOmega(mod)

#: Predict the future values (Exact and Gaussian approximation)
z_t = c(Y[,n],1,X[,n])
predExact = qExactPredictive(mod,qOm,z_t)
predGVApp = qApproxPredictive(mod,qOm,z_t)

#: Compare the two variational predictive densities
i = 1
plot(density(predExact[i,]),lwd=2,main=paste("gdl =",round(qOm$df),"and d =",d))
curve(dnorm(x,predGVApp$mu[i],sqrt(predGVApp$Sigma[i,i])),add=T,col=2,lwd=2)
legend("topleft",legend = c("Exact","Approx"), 
       col=c("black","red"),lty=1, lwd=2, cex=1.2)


