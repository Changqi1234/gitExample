args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
    cat("usage: Rscript hw4.R <template spectrum> <data directory>\n")
    quit(status = 1)
}

template_file <- args[1]
data_dir      <- args[2]

library("FITSio")

main   <- readFrameFromFITS(template_file)
n_main <- dim(main)[1]

list_of_origin <- list.files(data_dir)

record <- data.frame(
  distance   = numeric(),
  spectrumID = character(),
  i          = numeric(),
  stringsAsFactors = FALSE
)

for (i in seq_along(list_of_origin)) {
  data <- readFrameFromFITS(file.path(data_dir, list_of_origin[i]))
  n <- dim(data)

  if (n[2] %in% c(8, 9)) {
    data <- data[, 1:4]
  }

  times <- n[1] - n_main + 1

  if (times <= 0) {
    distance2 <- Inf
    place     <- NA
  } else {
    distance1 <- rep(NA_real_, times)

    for (j in 1:times) {
      data1 <- data[j:(j + n_main - 1), ]
      w <- (as.numeric(data1[, "and_mask"] == 0)) * data1[, "ivar"]
      flux1 <- (data1[, "flux"] - mean(data1[, "flux"])) / sd(data1[, "flux"])
      flux2 <- (main[,  "FLUX"] - mean(main[,  "FLUX"])) / sd(main[,  "FLUX"])
      diff <- flux1 - flux2
      distance1[j] <- sum(w * diff^2) / sum(w)
    }

    if (all(is.na(distance1))) {
      distance2 <- Inf
      place     <- NA
    } else {
      distance2 <- min(distance1, na.rm = TRUE)
      place     <- which.min(distance1)
    }
  }

  record <- rbind(
    record,
    data.frame(
      distance   = distance2,
      spectrumID = list_of_origin[i],
      i          = place,
      stringsAsFactors = FALSE
    )
  )
}

record_sorted <- record[order(record$distance), ]
print(record_sorted)

output_file <- paste0(data_dir, ".csv")
write.csv(record_sorted, file = output_file, row.names = FALSE)
