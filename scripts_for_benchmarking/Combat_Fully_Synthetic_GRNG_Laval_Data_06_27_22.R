rm(list=ls())
library(tidyverse)
library(sva)

exp_dir_1 = "~/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1/Data"
exp_dir_2 = paste(exp_dir_1, "Fully_Synthetic_Data_05_03_2023", sep="/")
exp_dir_3 = paste(exp_dir_1, "Fully_Synthetic_ComBat_Data_05_03_2023", sep="/")
folders = c("Uniform", "pi_over_four_kappa", "pi_over_two_kappa", "three_pi_over_four_kappa", "pi_kappa")
versions = c("V1", "V2", "V3", "V4", "V5")

make_file_names <- function(x, y) {
  file_names = rep("", 25)
  for (ii in 1:length(x)) {
    for (jj in 1:length(y)) {
      file_names[jj+(ii-1)*5] = paste(x[ii], paste(x[ii], paste(y[jj], "csv", sep="."), sep="_"), sep="/")
    }
  }
  return(file_names)
}

file_names = make_file_names(folders, versions)

test = readr::read_csv(paste(exp_dir_2, file_names[1], sep="/")) %>% as.data.frame
full_file = test
x = file_names[1]

do_combat_adjust <- function(x) {
  full_file = readr::read_csv(paste(exp_dir_2, x, sep="/")) %>% as.data.frame
  my_batches = as.factor(full_file[1,-1])
  expression_data = mutate_all(full_file[-1,-1], as.numeric)
  non_zero_expression = expression_data - min(expression_data) + 1
  my_logged_expression = log(non_zero_expression)
  ComBat_adjusted = ComBat(dat=my_logged_expression, batch=my_batches)
  unlogged = exp(ComBat_adjusted) + min(expression_data) - 1
  full_data_set = cbind(full_file[,1], rbind(full_file[1,-1], unlogged))
  colnames(full_data_set)[1] = "Gene_Symbols"
  readr::write_csv(full_data_set, paste(exp_dir_3, paste(substr(x, 1, nchar(x)-4), "ComBat_Adjusted.csv", sep = "_"), sep="/"))
}

sapply(file_names, do_combat_adjust)

