if (!require(muRty)) {
    quit("no", 2)
}

library(muRty)

# command args
temp_dir <- NULL

args <- commandArgs(trailingOnly=TRUE)
for (arg in args) {
    s <- strsplit(arg, "=")[[1]]
    if (length(s) < 2) {
        next
    }

    if (s[1] == "--temp_dir") {
        temp_dir <- paste(s[2:length(s)], collapse="=")
    }
}

# function args
path <- file.path(temp_dir, "X.csv")
X <- read.csv(path, sep=',', header=FALSE)

path <- file.path(temp_dir, "k.csv")
k <- read.csv(path, sep=',', header=FALSE)

# run muRty
result <- get_k_best(mat=as.matrix(X), k_best=k)

# write result
count = 0
for (i in seq_along(result$solutions)) {
    filename <- paste("solutions", sprintf("%08d", i), ".csv", sep="")
    path <- file.path(temp_dir, filename)
    write.csv(result$solutions[i], path, row.names=FALSE)
}

path <- file.path(temp_dir, "costs.csv")
write.csv(result$costs, path, row.names=FALSE)

quit("no", 0)
