suppressWarnings({ options(stringsAsFactors = FALSE) })
library(expm)

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 4) { stop("Usage: Rscript simulate_advecdiff.R <precomp.rds> <Dx> <Dy> <out_csv> [t_keep_csv]") }
precomp_rds <- args[1]; Dx <- as.numeric(args[2]); Dy <- as.numeric(args[3]); out_csv <- args[4]
t_keep <- ifelse(length(args)>=5, as.integer(args[5]), NA_integer_)

pre <- readRDS(precomp_rds)
T <- pre$T; delta <- pre$delta; N <- pre$N; K.set <- pre$K.set; F.inv <- pre$F.inv; y0 <- pre$y0; source <- pre$source;
Y0.R.source <- pre$Y0Rsource; Y0.I.source <-pre$Y0Isource; Y0.R <- pre$Y0R; Y0.I <-pre$Y0I;

theta1 <- 0.004; theta2 <- 0.008; theta4 <- 0.01
v <- c(theta1, theta2); D <- diag(c(Dx, Dy)); zeta <- theta4

# ------------------------------------------------------------

# prepare alpha
alpha.R.seq = alpha.I.seq = array(0/0,dim=c(N,T))
beta.R.seq = beta.I.seq = array(0/0,dim=c(N,1))

# set parameters
alpha.R.seq[,1] = Y0.R # pass initial condition
alpha.I.seq[,1] = Y0.I # pass initial condition
beta.R.seq[,1] = Y0.R.source # pass initial condition
beta.I.seq[,1] = Y0.I.source # pass initial condition

# beta0 <- matrix(c(Y0.R.source, Y0.I.source), ncol=1)

# Transition g
g.list = g.exp.list = list()
g.blk = array(0/0,dim=c(2,2))
for (i in 1:N){
  K = matrix(K.set[i,],ncol=1)
  tmp2 = -t(K)%*%D%*%K*4*pi^2 - zeta
  tmp1 = v%*%K*2*pi
  g.blk = matrix(c(tmp2, tmp1, -tmp1, tmp2),nrow=2)
  g.list[[i]] = g.blk
  g.exp.list[[i]] = expm(g.blk*delta)
}

# Backgound noise level
noise.R = array(0/0,dim=c(N,1))
noise.I = array(0/0,dim=c(N,1))
for (i in 1:N){
  noise.R[i] = abs(alpha.R.seq[i,1])/100
  noise.I[i] = abs(alpha.I.seq[i,1])/100
}

# transition iterations
for (i.t in 2:T){
  for (i in 1:N){
    
    #transition
    alpha.current = matrix( c(alpha.R.seq[i,i.t-1], alpha.I.seq[i,i.t-1]), ncol=1 )
    alpha.next = g.exp.list[[i]] %*% alpha.current # transition
    # add source
    beta0.current = matrix( c(beta.R.seq[i,1], beta.I.seq[i,1]), ncol=1 )
    alpha.next = alpha.next + beta0.current # adding source
    # add noise
    alpha.R.seq[i,i.t] = alpha.next[1] + rnorm(1,0,noise.R[i])
    alpha.I.seq[i,i.t] = alpha.next[2] + rnorm(1,0,noise.I[i])
  }
}

y = array(0/0,dim=c(N,T))
y[,1] = y0 # pass initial condition
for (i.t in 2:T){
  y[,i.t] = F.inv %*% matrix(c(alpha.R.seq[,i.t],alpha.I.seq[,i.t]),ncol=1)
}

if (!is.na(t_keep) && t_keep > 1) { sel <- unique(c(1, seq(1, T, by=t_keep), T)); y <- y[, sel, drop=FALSE] }
dir.create(dirname(out_csv), showWarnings=FALSE, recursive=TRUE)
write.table(y, file=out_csv, row.names=FALSE, col.names=FALSE, sep=",")
cat("Saved", out_csv, "\n")