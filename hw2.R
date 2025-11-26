library('FITSio')

main <- readFrameFromFITS("cB58_Lyman_break.fit")
n_main <- dim(main)[1]

list_of_origin <- list.files("data")
record <- data.frame(distance=numeric(),
                     spectrumID=character(),
                     i=numeric(),
                     stringsAsFactors=FALSE)

for (i in 1:100) {
  data <- readFrameFromFITS(paste0("./data/", list_of_origin[i]))
  n <- dim(data)

  if (n[2] %in% c(8,9)) {
    data <- data[,1:4]
  }

  times <- n[1] - n_main + 1
  

  distance1 <- numeric(times)

  for (j in 1:times) {
    data1 <- data[j:(j+n_main-1), ]
    w <- (as.numeric(data1[, 'and_mask']==0))*data1[,'ivar']
    flux1 <- (data1[, "flux"] - mean(data1[, "flux"])) / sd(data1[, "flux"])
    flux2 <- (main[, "FLUX"] - mean(main[, "FLUX"])) / sd(main[, "FLUX"])
    diff <- flux1 - flux2
    distance1[j] <- sum(w * diff^2)/sum(w)
  }
if (length(distance1)==0 || all(is.na(distance1))) {
    distance2 <- NA
    place <- NA
} else {
    distance2 <- min(distance1, na.rm=TRUE)
    place <- which.min(distance1)
}

  record <- rbind(record,
                  data.frame(distance=distance2,
                             spectrumID=list_of_origin[i],
                             i=place,
                             stringsAsFactors=FALSE))
}


record_sorted <- record[order(record$distance), ]
print(record_sorted)
write.csv(record_sorted, file="hw2.csv", row.names=FALSE)
