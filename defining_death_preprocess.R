# defining death, each hospital admission  is treated as a seperate entity
unclean_data <- read.csv("death_age_info.csv", na.strings=c("","NA"))

defining_death <- function(data, day){

  count <- 0
  dead <- list()
  dod <- data$dod_hosp
  out <- data$outtime
  home_dod <- data$death_date

  for (x in 1:nrow(data)){

    # patient did not die
    if (is.na(dod[x]) & is.na(home_dod[x])){
      count <- count + 1
      dead[x] <- 0
      next
    }
    out_1 <- strptime(out[x], "%Y-%m-%d %H:%M:%S")

    # home death
    if (is.na(dod[x]) & !is.na(home_dod[x])){
      dod_h <- strptime(home_dod[x], "%Y-%m-%d %H:%M:%S")
      difference <- difftime(dod_h, out_1, unit="day")

      if (difference <= day | is.na(difference)){
        dead[x] <- 1
        next
      } else {
        dead[x] <- 0
      }
      next
    }

    # hospital death
    dod_1 <- strptime(dod[x], "%Y-%m-%d %H:%M:%S")
    difference <- difftime(dod_1, out_1, unit="day")

    if (difference <= day | is.na(difference)){
      dead[x] <- 1
      next
    }else {
      dead[x] <- 0
    }
  }
  data$death <- unlist(dead)
  return(data)
}


unclean_data <- defining_death(unclean_data, 3)

unclean_data$subject_id <- NULL
unclean_data$death_date <- NULL
unclean_data$outtime <- NULL
unclean_data$dod_hosp <- NULL

write.csv(unclean_data, "death_within_3_days.csv", row.names=FALSE)