
# load data
data("College", package = "ISLR")
df <- College %>% 
  mutate(Private = ifelse(Private == "Yes", 1, 0))

write.csv(df, "./dat/College_editted.csv", row.names = F)

rm(list = ls())