suppressWarnings({ options(stringsAsFactors = FALSE) })
library(sp); library(expm); library(geoR); library(RPMG); library(VGAM)

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 1) { stop("Usage: Rscript simulate_precompute.R <precomp_out.rds>") }
precomp_out <- args[1]

T  <- 100
delta <- 0.1
x1 <- seq(0, 1-0.01, 0.02); x2 <- seq(0, 1-0.01, 0.02)
N1 <- length(x1); N2 <- length(x2); N <- N1 * N2
x1.rep <- rep(x1, each=N2); x2.rep <- rep(x2, N1); x.grid <- cbind(x1.rep, x2.rep)

set.seed(2)
n.Gaussian <- 10
center.x <- runif(n.Gaussian,0,1); center.y <- runif(n.Gaussian,0,1)

sim1 = grf(N, grid = "reg", cov.pars = c(1, .25))
y0 = matrix(sim1$data,ncol=1)+10
case =which(y0<0)
if (length(case)>0){y0[case]=0}

K1.seq = c((-N1/2+1):(N1/2))
K2.seq = c((-N1/2+1):(N1/2))
K1.set = rep(K1.seq, each=N2)
K2.set = rep(K2.seq, N1)
K.set = cbind(K1.set,K2.set)

Y0.R = Y0.I = array(0/0,dim=c(N,1))
for (j in 1:N){
  tmp1 = tmp2 = 0
  for (i in 1:N){
    s = matrix(x.grid[i,],ncol=1)
    tmp0 = K.set[j,] %*% s * 2 * pi
    tmp1 = tmp1 + y0[i,]*cos(tmp0)
    tmp2 = tmp2 + y0[i,]*sin(tmp0)
  }
  Y0.R[j] = tmp1/N
  Y0.I[j] = tmp2/N
}
F.inv = array(0/0,dim=c(N,2*N)) #this is a one-time computation
for (j in 1:N){
  for (i in 1:N){
    k = matrix(K.set[i,],ncol=1)
    tmp0 = x.grid[j,] %*% k * 2 * pi
    F.inv[j,i] = cos(tmp0)
    F.inv[j,i+N] = sin(tmp0)
  }
}

var.x = 0.005; var.y = 0.005
source  = array(0,dim=c(N,1))
for (i in 1:n.Gaussian){
  source =source + dbinorm(x.grid[,1], x.grid[,2], 
                           mean1 = center.x[i], mean2 = center.y[i], 
                           var1 = var.x, var2 = var.y, cov12 = 0,log = FALSE)
}
# source = source * 1/max(source)
source = source * 0/max(source)

Y0.R.source = Y0.I.source = array(0/0,dim=c(N,1))
for (j in 1:N){
  tmp1 = tmp2 = 0
  for (i in 1:N){
    s = matrix(x.grid[i,],ncol=1)
    tmp0 = K.set[j,] %*% s * 2 * pi
    tmp1 = tmp1 + source[i,]*cos(tmp0)
    tmp2 = tmp2 + source[i,]*sin(tmp0)
  }
  Y0.R.source[j] = tmp1/N
  Y0.I.source[j] = tmp2/N
}

pre <- list(T=T, delta=delta, N=N, x.grid=x.grid, K.set=K.set, F.inv=F.inv, y0=y0, source=source, Y0R=Y0.R, Y0I=Y0.I, Y0Rsource=Y0.R.source, Y0Isource=Y0.I.source)
saveRDS(pre, file=precomp_out)
cat("Saved precomp to", precomp_out, "\n")