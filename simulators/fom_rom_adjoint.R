suppressWarnings({ options(stringsAsFactors = FALSE) })
suppressPackageStartupMessages({ library(jsonlite) })

# Usage (两种):
# 1) 旧用法(保持兼容):
#    Rscript fom_rom_adjoint.R <precomp.rds> <phi_csv> <Dx> <Dy> <out_json> [t_keep] [fom_csv] [rom_csv] [noise_scale]
#    - <out_json> 为“具体文件路径”时，照旧写到该文件。
# 2) 新用法(推荐):
#    Rscript fom_rom_adjoint.R <precomp.rds> <phi_csv> <Dx> <Dy> <out_dir> [t_keep] ...
#    - 若 <out_dir> 是“目录”(存在或能创建)，将自动命名：
#      adj_r{r}_N{N}_Dx{...}_Dy{...}_f{...}_phi{md5_8}.json
#
# 说明：
# - 本脚本现在会在提供第7/8个参数时，额外写出 FOM/ROM 快照 CSV（矩阵尺寸 N x |keep_idx|）。
# - 由于 alpha = (1/N)*[C^T; S^T] y 的定义，推导得到 A_r 及其导数均应除以 N，因此保留 /N。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop("Usage: Rscript fom_rom_adjoint.R <precomp.rds> <phi_csv> <Dx> <Dy> <out_json|out_dir> [t_keep] [fom_csv] [rom_csv] [noise_scale]")
}
precomp_rds <- args[[1]]
phi_csv     <- args[[2]]
Dx          <- as.numeric(args[[3]])
Dy          <- as.numeric(args[[4]])
out_arg     <- args[[5]]
t_keep      <- if (length(args) >= 6 && nzchar(args[[6]])) as.integer(args[[6]]) else NA_integer_
fom_csv     <- if (length(args) >= 7 && nzchar(args[[7]])) args[[7]] else NA_character_
rom_csv     <- if (length(args) >= 8 && nzchar(args[[8]])) args[[8]] else NA_character_
noise_scale <- if (length(args) >= 9 && nzchar(args[[9]])) as.numeric(args[[9]]) else 0.0

# ---------- load precomputed ----------
pre <- readRDS(precomp_rds)
T     <- pre$T
delta <- pre$delta
N     <- pre$N
K.set <- pre$K.set
F.inv <- pre$F.inv
y0    <- pre$y0
source_vec <- pre$source
Y0.R  <- pre$Y0R
Y0.I  <- pre$Y0I
Y0.R.src <- pre$Y0Rsource
Y0.I.src <- pre$Y0Isource

# constants
theta1 <- 0.004; theta2 <- 0.008; theta4 <- 0.01
v <- c(theta1, theta2); zeta <- theta4

# ---------- POD basis ----------
Phi <- as.matrix(read.csv(phi_csv, header = FALSE))
if (nrow(Phi) != N) stop("Phi row count != N in precomp.")
r <- ncol(Phi); if (r < 1) stop("Phi must have at least one column.")

# convenience matrices
Cmat <- F.inv[, 1:N, drop=FALSE]
Smat <- F.inv[, (N+1):(2*N), drop=FALSE]

# ROM ingredients
x0  <- as.numeric(t(Phi) %*% y0)          # r
U_r <- as.numeric(t(Phi) %*% source_vec)  # r

V_R <- crossprod(Cmat, Phi)       # N x r
V_I <- crossprod(Smat, Phi)       # N x r

# ---------- per-mode coefficients ----------
pi2 <- 2*pi
TWOPI_SQ <- 4*pi^2
Kx <- K.set[,1]; Ky <- K.set[,2]
tmp1 <- (v[1]*Kx + v[2]*Ky) * pi2
tmp2 <- -(Dx*(Kx*Kx) + Dy*(Ky*Ky)) * (TWOPI_SQ) - zeta

expfac <- exp(delta * tmp2)
cang <- cos(delta * tmp1); sang <- sin(delta * tmp1)
a_vec <- expfac * cang                # length N
b_vec <- expfac * sang

# d tau / d p → s_i = dt * d tau / d p
sDx <- delta * (-(TWOPI_SQ) * (Kx*Kx))
sDy <- delta * (-(TWOPI_SQ) * (Ky*Ky))
aDx_vec <- sDx * a_vec
bDx_vec <- sDx * b_vec
aDy_vec <- sDy * a_vec
bDy_vec <- sDy * b_vec

# ---------- build A_r and its derivatives ----------
# 公式: A = VR^T Diag(a) VR + VI^T Diag(a) VI + VR^T Diag(b) VI - VI^T Diag(b) VR
# 用“按行加权”，避免显式 diag(N)
matmul_weighted <- function(A, w, B) {      # returns t(A) %*% (diag(w) %*% B)
  # 等价于 t(A) %*% (B * w)  (B 的每一行乘以 w_i)
  return( t(A) %*% (B * w) )
}
A_r   <- matmul_weighted(V_R, a_vec,   V_R) +
         matmul_weighted(V_I, a_vec,   V_I) +
         matmul_weighted(V_R, b_vec,   V_I) -
         matmul_weighted(V_I, b_vec,   V_R)

dArDx <- matmul_weighted(V_R, aDx_vec, V_R) +
         matmul_weighted(V_I, aDx_vec, V_I) +
         matmul_weighted(V_R, bDx_vec, V_I) -
         matmul_weighted(V_I, bDx_vec, V_R)

dArDy <- matmul_weighted(V_R, aDy_vec, V_R) +
         matmul_weighted(V_I, aDy_vec, V_I) +
         matmul_weighted(V_R, bDy_vec, V_I) -
         matmul_weighted(V_I, bDy_vec, V_R)

# 重要：由于 alpha = (1/N)*[C^T; S^T] y，推导得到 A_r 与 dA_r/dp 都需除以 N
A_r   <- A_r   / N
dArDx <- dArDx / N
dArDy <- dArDy / N

# ---------- time selection ----------
keep_idx <- 1:T
if (!is.na(t_keep) && t_keep > 1) {
  keep_idx <- sort(unique(c(1, seq(1, T, by=t_keep), T)))
}

# ---------- forward pass: deterministic ----------
alphaR <- matrix(0.0, nrow=N, ncol=T)
alphaI <- matrix(0.0, nrow=N, ncol=T)
alphaR[,1] <- as.numeric(Y0.R)
alphaI[,1] <- as.numeric(Y0.I)
betaR <- as.numeric(Y0.R.src)
betaI <- as.numeric(Y0.I.src)

for (tt in 2:T) {
  alphaR_prev <- alphaR[,tt-1]; alphaI_prev <- alphaI[,tt-1]
  alphaR[,tt] <- a_vec * alphaR_prev + b_vec * alphaI_prev + betaR
  alphaI[,tt] <- -b_vec * alphaR_prev + a_vec * alphaI_prev + betaI
}

x <- matrix(0.0, nrow=r, ncol=T)
x[,1] <- x0
for (tt in 2:T) {
  x[,tt] <- A_r %*% x[,tt-1] + U_r
}

# ---------- residuals & objective ----------
f_half <- 0.0
r_list <- vector("list", T)
for (tt in keep_idx) {
  y_f <- Cmat %*% alphaR[,tt,drop=FALSE] + Smat %*% alphaI[,tt,drop=FALSE]   # N x 1
  y_r <- Phi  %*% x[,tt,drop=FALSE]
  rv  <- as.numeric(y_f - y_r)
  f_half <- f_half + 0.5 * sum(rv * rv)
  r_list[[tt]] <- rv
}

# ---------- adjoint ----------
lamR <- matrix(0.0, nrow=N, ncol=T)
lamI <- matrix(0.0, nrow=N, ncol=T)
mu   <- matrix(0.0, nrow=r, ncol=T)

if (T %in% keep_idx) {
  rT <- r_list[[T]]
  lamR[,T] <- as.numeric(t(Cmat) %*% rT)
  lamI[,T] <- as.numeric(t(Smat) %*% rT)
  mu[,T]   <- as.numeric(-t(Phi)  %*% rT)
}
for (tt in seq(T, 2, by=-1)) {
  # ROM adjoint
  mu_prev <- as.numeric(t(A_r) %*% mu[,tt])
  if ((tt-1) %in% keep_idx) {
    rprev <- r_list[[tt-1]]
    mu_prev <- mu_prev - as.numeric(t(Phi) %*% rprev)
  }
  mu[,tt-1] <- mu_prev
}
for (tt in seq(T, 2, by=-1)) {
  # FOM adjoint
  lamR_prev <-  a_vec * lamR[,tt] - b_vec * lamI[,tt]
  lamI_prev <-  b_vec * lamR[,tt] + a_vec * lamI[,tt]
  if ((tt-1) %in% keep_idx) {
    rprev <- r_list[[tt-1]]
    lamR_prev <- lamR_prev + as.numeric(t(Cmat) %*% rprev)
    lamI_prev <- lamI_prev + as.numeric(t(Smat) %*% rprev)
  }
  lamR[,tt-1] <- lamR_prev
  lamI[,tt-1] <- lamI_prev
}

# ---------- gradient (FOM + ROM 部分) ----------
accum_FOM <- function(s_p) {
  # sum_{t=2..T} s_p · [ lamR_t ⊙ vR_{t-1} + lamI_t ⊙ vI_{t-1} ]
  acc <- 0.0
  for (tt in 2:T) {
    vR <- a_vec * alphaR[,tt-1] + b_vec * alphaI[,tt-1]
    vI <- -b_vec * alphaR[,tt-1] + a_vec * alphaI[,tt-1]
    acc <- acc + sum( s_p * ( lamR[,tt] * vR + lamI[,tt] * vI ) )
  }
  acc
}
g_Dx <- accum_FOM(sDx) + sum( sapply(2:T, function(tt) as.numeric( t(mu[,tt]) %*% (dArDx %*% x[,tt-1]) ) ) )
g_Dy <- accum_FOM(sDy) + sum( sapply(2:T, function(tt) as.numeric( t(mu[,tt]) %*% (dArDy %*% x[,tt-1]) ) ) )

# ---------- robust checks ----------
finite_all <- function(v) { all(is.finite(v)) && !any(is.nan(v)) }
if (!finite_all(f_half)) stop("f is not finite")
if (!finite_all(g_Dx) || !finite_all(g_Dy)) {
  stop(sprintf("Non-finite grad: gDx=%.3e gDy=%.3e", g_Dx, g_Dy))
}

# ---------- (新增) 可选写出 FOM/ROM 快照 CSV ----------
if (!is.na(fom_csv) || !is.na(rom_csv)) {
  if (!is.na(fom_csv)) dir.create(dirname(fom_csv), recursive = TRUE, showWarnings = FALSE)
  if (!is.na(rom_csv)) dir.create(dirname(rom_csv), recursive = TRUE, showWarnings = FALSE)
  cols <- keep_idx  # 与 Python 侧一致：导出保留时刻

  # FOM: y_f = C*alphaR + S*alphaI   维度 N x |cols|
  if (!is.na(fom_csv)) {
    Yf_mat <- Cmat %*% alphaR[, cols, drop=FALSE] + Smat %*% alphaI[, cols, drop=FALSE]
    write.table(Yf_mat, file = fom_csv, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
    cat(sprintf("[snapshots] wrote FOM to %s (%d x %d)\n", fom_csv, nrow(Yf_mat), ncol(Yf_mat)))
  }

  # ROM: y_r = Phi * x               维度 N x |cols|
  if (!is.na(rom_csv)) {
    Yr_mat <- Phi %*% x[, cols, drop=FALSE]
    write.table(Yr_mat, file = rom_csv, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
    cat(sprintf("[snapshots] wrote ROM to %s (%d x %d)\n", rom_csv, nrow(Yr_mat), ncol(Yr_mat)))
  }
}

# ---------- write JSON with deterministic naming (可选目录模式) ----------
dir.create(dirname(out_arg), showWarnings = FALSE, recursive = TRUE)
phi_md5 <- as.character(tools::md5sum(phi_csv))
fmt_token <- function(x) {            # 便于做文件名: - → m, . → p
  s <- sprintf("%.8g", x)
  s <- gsub("-", "m", s, fixed=TRUE)
  s <- gsub("\\.", "p", s)
  s
}
# 若 out_arg 是目录或以 / 结尾，则自动命名
is_dir_like <- function(p) {
  has_slash <- grepl("/$", p)
  is_dir <- FALSE
  if (file.exists(p)) is_dir <- file.info(p)$isdir
  if (!file.exists(p) && has_slash) {
    dir.create(p, recursive=TRUE, showWarnings=FALSE)
    is_dir <- TRUE
  }
  is_dir || has_slash
}

f_val <- f_half
payload <- list(
  f = unbox(f_val),
  grad = I(c(g_Dx, g_Dy)),
  point = list(Dx = unbox(Dx), Dy = unbox(Dy)),
  basis = list(r = unbox(r), N = unbox(N), phi_csv = normalizePath(phi_csv), phi_md5 = unbox(phi_md5)),
  time = list(T = unbox(T), delta = unbox(delta), kept = I(as.integer(keep_idx))),
  meta = list(precomp = normalizePath(precomp_rds),
              script = "fom_rom_adjoint.R",
              version = "1.2.0")
)

if (is_dir_like(out_arg)) {
  fname <- sprintf("adj_r%d_N%d_Dx%s_Dy%s_f%s_phi%s.json",
                   r, N, fmt_token(Dx), fmt_token(Dy), fmt_token(f_val), substr(phi_md5, 1, 8))
  final_path <- file.path(out_arg, fname)
} else {
  final_path <- out_arg
  dir.create(dirname(final_path), showWarnings = FALSE, recursive = TRUE)
}
cat(toJSON(payload, auto_unbox = TRUE), file = final_path)
cat("\n")
cat(sprintf("[adjoint] point=(Dx=%.10g, Dy=%.10g)  r=%d  out=%s\n",
            Dx, Dy, r, final_path))

if (noise_scale > 0) {
  message("NOTE: noise_scale was provided but is ignored to keep gradients deterministic.")
}

# ---------- diagnostics: MSE/RMS & scaled grads ----------
n_keep <- length(keep_idx)
mse <- f_half / (N * n_keep)
rms <- sqrt(2 * mse)
cat(sprintf("Per-point RMS residual = %.3f\n", rms))
cat(sprintf("Scaled grads: dMSE/dDx = %.3e, dMSE/dDy = %.3e\n",
            g_Dx/(N*n_keep), g_Dy/(N*n_keep)))
cat(sprintf("Grad log-params: dMSE/d(log Dx) = %.3e, dMSE/d(log Dy) = %.3e\n",
            (g_Dx*Dx)/(N*n_keep), (g_Dy*Dy)/(N*n_keep)))
