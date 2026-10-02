# Hazelton (2024) R reference for P1, from his supplementary MaxEdgeLoading/SLEM.
# Emits results/reference_R.txt so the Python core can be cross-checked bit-for-bit.
# Run:  Rscript experiments/R_reference.R
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
if (length(here) == 0) here <- "."
root <- normalizePath(file.path(here, ".."))
source(file.path(root, "reference", "hazelton_functions.R"))

A  <- rbind(c(1,1,1,0,0,0,1,0), c(0,1,1,0,0,1,0,1),
            c(0,0,1,0,1,0,0,1), c(0,0,0,1,0,0,1,0))
A2 <- A[, c(7,1,6,5,2,3,8,4)]                 # Example 7 reordering (Markov basis)
U2 <- LatticeBasis(A2, reorder = FALSE)$U
yy <- c(A %*% c(1,1,1,1,0,1,1,1))             # = (4,4,2,2)
ord <- c(7,1,6,5,2,3,8,4)

out <- file(file.path(root, "results", "reference_R.txt"), "w")
writeLines(sprintf("# Hazelton SM-code.r reference (P1). fibre size = %d",
                   ncol(FindFibre(A2, yy))), out)
ug <- 1 - SLEM(A2, U2, y = yy, UNIFORM = TRUE)$SLEM
ur <- MaxEdgeLoading(A2, U2, y = yy, UNIFORM = TRUE)$MaxEdgeLoading
writeLines(sprintf("uniform   gap=%.8f  rho=%.6f", ug, ur), out)
writeLines("theta5    SLEM         gap            rho", out)
for (t5 in c(1, 0.5, 0.2, 0.1, 0.05, 0.01)) {
  lam <- c(1,1,1,1,t5,1,1,1)
  sg  <- SLEM(A2, U2, y = yy, lam[ord], UNIFORM = FALSE)$SLEM
  ml  <- MaxEdgeLoading(A2, U2, y = yy, lambda = lam[ord], UNIFORM = FALSE)
  writeLines(sprintf("%-8g  %.8f   %.8e   %.6f", t5, sg, 1 - sg, ml$MaxEdgeLoading), out)
}
close(out)
cat("wrote results/reference_R.txt\n")
