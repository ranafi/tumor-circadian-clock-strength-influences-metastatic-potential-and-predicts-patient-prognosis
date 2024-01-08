# Start script ####

rm(list = ls()) # Clear any global variables

# Load Libraries ####

library(tidyverse) # Load the tidyverse package
library(aod)

# Load Source Files ####

gtex_plotting_function_file_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Reusable_R_Functions/GTEX_TCGA_UK_Plotting_Functions.R"
source(gtex_plotting_function_file_path) # Load
Load_Data_Helper_function_file_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Reusable_R_Functions/Load_Data_Helper_Functions.R"
source(Load_Data_Helper_function_file_path) # Load
Nested_GLM_Test_file_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Reusable_R_Functions/Nested_GLM_Test.R"
source(Nested_GLM_Test_file_path) # Load

# Load Sample Fit ####

fit_path = "~/Library/CloudStorage/Box-Box/Share_Jan_and_Ron/GTEx_TCGA_UK/TNT_Training/training_results/NTLA_2021-12-16T10_31_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_0/Luminal_A/2022-01-11T12_21_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_0/Fits"
fit_files = load_data(fit_path) # Load CYCLOPS cosine fit and CYCLOPS sample fit output files
sample_fit = fit_files$fit_file %>% make_sample_phase_magnitude_data_frame_group # gather only CYCLOPS sample fit output file

# TCGA Covariates ####

{
## Load TCGA Covariates ####

base_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Data"
tcga_uk_covariates_path = paste(base_path, "brcaUK_202008_TCGA-BRCA_Covariates.csv", sep = "/")
load_tcga_uk_covariates = readr::read_csv(tcga_uk_covariates_path) # Load TCGA and UK covariates file

# Clean TCGA Covariates ####

tcga_uk_covariates = load_tcga_uk_covariates %>%
  column_to_rownames("Gene.Symbol") %>% # make the column mislabled "Gene.Symbols" the row names. This column actually contains sample IDs.
  t %>% # transpose to make sample instances go from columns to rows
  as.data.frame %>% # convert tibble to data frame
  dplyr::mutate(age = as.numeric(age)) %>% # change age from type character to number
  rownames_to_column(var = "ID") # make a column from data frame row names named "ID"

# Combine Sample Fit with TCGA Covariates ####

sample_fit = sample_fit %>% # Update CYCLOPS sample output...
  left_join(tcga_uk_covariates, by = "ID") %>% # ...by merging the TCGA and UK covariates...
  mutate(Mag_Group_Low_Med_merged = Magnitude_Group != "High") %>% # ...and creating a new magnitude group classification which merges the low and medium magnitude samples...
  mutate(Mag_Group_Low_High_merged = Magnitude_Group != "Med") # ...and an alternative magnitude group classification which merges the low and high magnitude samples.

}

# Clinical Sample Data ####

{
## Load Clinical Sample Data ####

clinical_file_path = "~/Library/CloudStorage/Box-Box/Share_Jan_and_Ron/GTEx_TCGA_UK/Broad_Institute_FireBrowse/Files_Created_From_Clinical_Files/TCGA_Clinical_Adverse_Events.csv"
load_clinical = readr::read_csv(clinical_file_path) # Load clinical TCGA data

# Clean Clinical Sample Data ####

clinical = load_clinical %>% # Update clinical TCGA data...
  as.data.frame %>% # ...convert tibble to data frame...
  t %>% # ...transpose data frame...
  as.data.frame %>% # ...convert back to data frame after transpose...
  'colnames<-'(.[1,]) %>% # ...make the column names of the data frame the first row of the data frame...
  dplyr::slice(-1) %>% # ...remove the first row...
  rownames_to_column(var = "ID_temp") %>% # ...make the row names of the data frame (sample IDs) a temporary column...
  dplyr::mutate(ID = substring(ID_temp, 1, 12)) %>% # ...make a permanent column for sample IDs, which is the first 12 characters of the temporary IDs...
  dplyr::relocate("ID") %>% # ...move sample ID column to the first column of the data frame...
  column_to_rownames(var = "ID_temp") # ...use the temporary column to become the row names of the data frame again.

# Clinical "pathologic" categories ####

{
## List of pathologic n categories ####

clinical %>% dplyr::select(contains("pathologic_n")) %>% unlist %>% unique
# [1]  "n0"        "n0 (i-)"   "n1"        "n1a"       "n2a"       "n3a"       "n3"        "n1c"       "n1mi"     
# [10] "n2"        "n1b"       "nx"        "n0 (i+)"   "n0 (mol+)"

# List of pathologic m categories ####
clinical %>% dplyr::select(contains("pathologic_m")) %>% unlist %>% unique
# [1]  "m0"       "mx"       "cm0 (i+)" "m1"

## Samples in Clinical categories ####

{
### Multiple Node Infiltration at Diagnosis ####

{
#### Sample IDs ####  

sample_ids_with_node_infiltration_at_diagnosis = clinical %>%
  dplyr::select(contains("pathologic_n")) %>%
  dplyr::filter(if_any(contains("pathologic_n"), ~ . %in% c("n2", "n2a", "n3", "n3a"))) %>%
  row.names %>%
  substring(1, 12) %>%
  unique

### Add Logical Vector ####

sample_fit = sample_fit %>%
  dplyr::mutate(Node.Infiltration.At.Diagnosis = case_when(
    substring(ID, 1, 12) %in% sample_ids_with_node_infiltration_at_diagnosis ~ T,
    .default = F))

}

## Metastasis at Diagnosis ####

{
#### Sample IDs ####

sample_with_metastasis_at_diagnosis = clinical %>%
  dplyr::select(contains("pathologic_m")) %>%
  dplyr::filter(if_any(contains("pathologic_m"), ~ . %in% c("m1", "cm0 (i+)"))) %>%
  row.names %>%
  substring(1, 12) %>%
  unique

### Add Logical Vector ####

sample_fit = sample_fit %>%
  dplyr::mutate(Metastasis.At.Diagnosis = case_when(
    substring(ID, 1, 12) %in% sample_with_metastasis_at_diagnosis ~ T,
    .default = F))

}

## Died within 5 Years of Diagnosis ####

{
#### Sample IDs ####

samples_that_died_within_5_years_of_diagnosis = clinical %>%
  dplyr::select(contains("death")) %>% # gather only columns with "death" in the column name
  dplyr::filter_at(vars(everything()), any_vars(!is.na(.))) %>% # keep only rows where at least one of the columns is no <NA>
  tidyr::unite(united_days_to_death, na.rm = T) %>% # combine all columns into one column
  dplyr::mutate(united_days_to_death = str_extract(string = united_days_to_death, pattern = "[0-9]{1,}")) %>% # extract the number of days till death
  dplyr::mutate(united_days_to_death = as.numeric(united_days_to_death)) %>% # convert extracted strings to numbers
  dplyr::mutate(years_to_death = united_days_to_death/365) %>% # create a new column for the number of years to death (number to days to death divided by 365)
  dplyr::filter(years_to_death <= 5) %>% # keep only rows for which years to death is less than or equal to 5
  row.names %>% # extract the row names, which are the subject IDs
  substring(1, 12) %>%
  unique

### Add Logical Vector ####

sample_fit = sample_fit %>%
  dplyr::mutate(Died.Within.5.Years.Of.Diagnosis = case_when(
    substring(ID, 1, 12) %in% samples_that_died_within_5_years_of_diagnosis ~ T,
    .default = F))

}

## New Tumor Events within 5 Years of Diagnosis ####

{
#### Sample IDs ####

samples_with_new_tumor_event_within_5_years = clinical %>% 
  dplyr::select(contains("new_tumor")) %>% # gather only columns with "new_tumor" in the column name
  dplyr::filter_at(vars(everything()), any_vars(!is.na(.))) %>% # keep only rows where at least one of the columns is no <NA>
  tidyr::unite(united_days_to_new_tumor_event, na.rm = T) %>% # combine all columns into one column
  dplyr::mutate(united_days_to_new_tumor_event = str_extract(string = united_days_to_new_tumor_event, pattern = "[0-9]{1,}")) %>% # extract the number of days till metastasis
  dplyr::mutate(united_days_to_new_tumor_event = as.numeric(united_days_to_new_tumor_event)) %>% # convert extracted strings to numbers
  dplyr::mutate(years_to_new_tumor_event = united_days_to_new_tumor_event/365) %>% # create a new column for the number of years to metastasis (number to days to metastasis divided by 365)
  dplyr::filter(years_to_new_tumor_event <= 5) %>% # keep only rows for which years to metastasis is less than or equal to 5
  row.names # extract the row names, which are the subject IDs

### Add Logical Vector ####

sample_fit = sample_fit %>%
  dplyr::mutate(New.Tumor.Event.Within.5.Years = case_when(
    substring(ID, 1, 12) %in% samples_with_new_tumor_event_within_5_years ~ T,
    .default = F))

}

## Any Adverse Event at Diagnosis ####

{
#### Add Logical Vector ####

sample_fit = sample_fit %>%
  dplyr::mutate(Adverse.Event.At.Diagnosis = ifelse(if_any(contains("At.Diagnosis"), ~ .), 
                                                    T, 
                                                    F))
}

## Any Adverse Event Within 5 Years of Diagnosis ####

{
#### Add Logical Vector ####

sample_fit = sample_fit %>%
  dplyr::mutate(Adverse.Event.Within.5.Years = ifelse(if_any(Died.Within.5.Years.Of.Diagnosis:New.Tumor.Event.Within.5.Years, ~ .), 
                                                      T, 
                                                      F))
}

## Any Adverse Event ####

{
#### Add Logical Vector ####

sample_fit = sample_fit %>%
  mutate(Any.Adverse.Event = ifelse(if_any(Node.Infiltration.At.Diagnosis:New.Tumor.Event.Within.5.Years, ~ .), 
                                    T, 
                                    F))
}
}
}
}

# MKI67 and ARNTL Expression ####
{
## Load Expression File ####
{
expression_data_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Figure3/Tumor_Sample_Magnitude/files/Expression_Data/gtex_tcga_uk_non_tumor_and_tumor_data-batches_with_more_than_5_samples.csv"
expression_data_file = readr::read_csv(expression_data_path) %>% as.data.frame
}

# Add MKI67 and ARNTL ####
{
MKI67_index = grep("MKI67$", expression_data_file$Gene.Symbol)
ARNTL_index = grep("ARNTL$", expression_data_file$Gene.Symbol)
sample_fit = sample_fit %>%
  dplyr::mutate(MKI67 = expression_data_file[MKI67_index, sample_fit$ID] %>%
                  mutate_all(as.numeric) %>%
                  unlist(use.names = F)) %>%
  dplyr::mutate(ARNTL = expression_data_file[ARNTL_index, sample_fit$ID] %>%
                  mutate_all(as.numeric) %>%
                  unlist(use.names = F))
}
# Make Tertiles of Expression ####
{

}
# Add Tertiles ####
}
# Single Independent Variable Anova ####

{
## Age ####
{
### LM ####

{  
#### Model ####

lm_age = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age)

### ANOVA ####

aov(data    = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age) %>%
  summary
# p-value = 0.00171

}

## GLM ####

{
### Model ####

glm_age = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age,
      family  = "binomial") %>%
  summary
# age p-value = 0.003
# AIC = 108.95

### ChiSq ####

get_glm_p_value(glm_age)
# p-value = 0.001475412

}
}

# MKI67 Expression ####
{
## LM ####

{
### Model ####

lm_MKI67 =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ MKI67)

## ANOVA ####
aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ MKI67) %>%
  summary
# p-value = 0.05
}

# GLM ####

{
### Model ####

glm_MKI67 =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ MKI67,
      family = "binomial") %>%
  summary
# MKI67 p-value = 0.0576
# AIC = 118.99

## ChiSq ####

get_glm_p_value(glm_MKI67)
# p-value = 0.07197562

}
}

# ARNTL Expression ####
{
## LM ####

{
### Model ####

lm_ARNTL =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ ARNTL)

## ANOVA ####
aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ ARNTL) %>%
  summary
# p-value = 0.635
}

# GLM ####

{
### Model ####
glm_ARNTL =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ ARNTL,
      family = "binomial") %>%
  summary
# ARNTL p-value = 0.633
# AIC = 122.01

## ChiSq ####
get_glm_p_value(glm_ARNTL)
# p-value = 0.6392358

}
}

# Magnitude Group ####
{
## LM ####

{
### Model ####

lm_magnitude_group = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Magnitude_Group)

## ANOVA ####
aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Magnitude_Group) %>%
  summary
# p-value = 0.0469
}

# GLM ####

{
### Model ####

glm_magnitude_group = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ Magnitude_Group,
      family  = "binomial") %>%
  summary
# Low.Magnitude.Group p-value = 3.26e-08
# Med.Magnitude.Group p-value = 0.261
# High.Magnitude.Group p-value = 0.189
# AIC = 117.91

## ChiSq ####

get_glm_p_value(glm_magnitude_group)
# p-value = 0.04252393

}
}

# Metastasis at Diagnosis ####
{
### LM ####

{
#### Model ####

lm_metastasis_at_diagnosis = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Metastasis.At.Diagnosis)

### ANOVA ####
aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Metastasis.At.Diagnosis) %>%
  summary
# p-value = 0.00175
}

## GLM ####

{
### Model ####

glm_metastasis_at_diagnosis =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ Metastasis.At.Diagnosis,
      family  = "binomial") %>% 
  summary
# No.Metastasis.At.Diagnosis p-value = <2e-16
# Metastasis.At.Diagnosis p-value = 0.0138
# AIC = 117.17

### ChiSq ####

get_glm_p_value(glm_metastasis_at_diagnosis)
# p-value = 0.02455663

}
}

# Multiple Node Infiltration at Diagnosis ####
{
## LM ####

{
### Model ####

lm_multiple_node_infiltration_at_diagnosis = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Node.Infiltration.At.Diagnosis)

## ANOVA ####

aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Node.Infiltration.At.Diagnosis) %>%
  summary
# p-value = 0.195

}

## GLM ####

{
### Model ####

glm_node_infiltration_at_diagnosis =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ Node.Infiltration.At.Diagnosis,
      family  = "binomial") %>%
  summary
# No.Node.Infiltration.At.Diagnosis p-value = <2e-16
# Node.Infiltration.At.Diagnosis p-value = 0.203
# AIC = 120.78
  
## ChiSq ####

get_glm_p_value(glm_node_infiltration_at_diagnosis)
# p-value = 0.2289104

}
}

# Any Adverse Event at Diagnosis ####
{
## LM ####

{  
### Model ####

lm_adverse_event_at_diagnosis = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Adverse.Event.At.Diagnosis)

## ANOVA ####

aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ Adverse.Event.At.Diagnosis) %>%
  summary
# p-value = 0.0744

}

## GLM ####

{
### Model ####

glm_adverse_event_at_diagnosis = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ Adverse.Event.At.Diagnosis,
      family  = "binomial") %>%
  summary
# No.Adverse.Event.At.Diagnosis p-value = <2e-16
# Adverse.Event.At.Diagnosis p-value = 0.0836
# AIC = 119.57

## ChiSq ####

get_glm_p_value(glm_adverse_event_at_diagnosis)
# p-value = 0.1029463

}
}

# New Tumor Event within 5 Years of Diagnosis ####
{
## LM ####

{
### Model ####

lm_new_tumor_event_within_5_years_of_diagnosis = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ New.Tumor.Event.Within.5.Years)

## ANOVA ####

aov(data = sample_fit,
    formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ New.Tumor.Event.Within.5.Years) %>%
  summary
# p-value = 0.000847

}

## GLM ####
  
{
### Model ####

glm_new_tumor_event_within_5_years_of_diagnosis =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ New.Tumor.Event.Within.5.Years,
      family  = "binomial") %>%
  summary
# No.New.Tumor.Event.Within.5.Years p-value = < 2e-16
# New.Tumor.Event.Within.5.Years p-value = 0.00362
# AIC = 115.35

## ChiSq ####

get_glm_p_value(glm_new_tumor_event_within_5_years_of_diagnosis)
# p-value = 0.00871315

}
}
}

# Non-Magnitude Nested Models ####

{
## Age and Adverse Event at Diagnosis ####
{
### LM ####
  
{
#### Model ####

lm_age_and_adverse_event_at_diagnosis =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Adverse.Event.At.Diagnosis)

### ANOVA ####
anova_death_by_adverse_event_at_diagnosis_on_top_of_age = 
  anova(lm_age,
        lm_age_and_adverse_event_at_diagnosis)
# p-value = 0.08728

}
  
## GLM ####

{
### Model ####
glm_age_and_adverse_event_at_diagnosis =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Adverse.Event.At.Diagnosis,
      family  = "binomial") %>%
  summary
# age p-value = 0.00205
# No.Adverse.Event.At.Diagnosis p-value = 1.06e-05
# Adverse.Event.At.Diagnosis p-value = 0.06379
# AIC = 107.83
  
## ChiSq ####

get_nested_glm_p_value(glm_age_and_adverse_event_at_diagnosis, glm_age)
# p-value = 0.07733241

}
}

# Age and MKI67 ####
{
### LM ####

{
#### Model ####

lm_age_and_MKI67 =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + MKI67)

### ANOVA ####
anova_death_by_MKI67_on_top_of_age = 
  anova(lm_age,
        lm_age_and_MKI67)
# p-value = 0.01351
}

## GLM ####

{
#### Model ####

glm_age_and_MKI67 =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + MKI67,
      family = "binomial") %>%
  summary
# age p-value = 0.000748
# MKI67 p-value = 0.008134
# AIC 104.46

### ChiSq ####

get_nested_glm_p_value(glm_age_and_MKI67, glm_age)
# p-value = 0.01084725

}
}

# Age and ARNTL ####
{
### LM ####

{
#### Model ####
lm_age_and_ARNTL =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + ARNTL)

### ANOVA ####
anova_death_by_ARNTL_on_top_of_age =
  anova(lm_age,
        lm_age_and_ARNTL)
# p-value = 0.5149
}

## GLM ####
{
#### Model ####
glm_age_and_ARNTL =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + ARNTL,
      family = "binomial") %>%
  summary
# age p-value = 0.00263
# ARNTL p-value = 0.45698
# AIC = 110.42

### ChiSq ####

get_nested_glm_p_value(glm_age_and_ARNTL, glm_age)
# p-value = 0.46775833
}
}
# Age and Metastasis at Diagnosis ####
{
## LM ####

{
### Model ####

lm_age_and_metastasis_at_diagnosis =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis)
  
## ANOVA ####

anova_death_by_metastasis_at_diagnosis_on_top_of_age = 
  anova(lm_age, 
        lm_age_and_metastasis_at_diagnosis)
# p-value = 0.0008932
  
}

## GLM ####

{
### Model ####

glm_age_and_metastasis_at_diagnosis =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis,
      family  = "binomial") %>%
  summary
# age p-value = 0.000983
# Metastasis.At.Diagnosis p-value = 0.001813
# No.Metastasis.At.Diagnosis p-value = 9.05e-06
# AIC = 102.9
  
## ChiSq ####

get_nested_glm_p_value(glm_age_and_metastasis_at_diagnosis, glm_age)
# p-value = 0.004546082

}
}

# Age, Metastasis at Diagnosis and MKI67 ####
{
## LM ####

{
### Model ####

lm_age_metastasis_at_diagnosis_and_MKI67 =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + MKI67)

## ANOVA ####
anova_death_by_MKI67_on_top_of_age_and_metastasis_at_diagnosis =
  anova(lm_age_and_metastasis_at_diagnosis,
        lm_age_metastasis_at_diagnosis_and_MKI67)
# p-value = 0.02094
}

## GLM ####

{
### Model ####

glm_age_metastasis_at_diagnosis_and_MKI67 =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + MKI67,
      family = "binomial") %>%
  summary
# age p-value = 0.000331
# Metastasis.At.Diagnosis p-value = 0.005546
# No.Metastasis.At.Diagnosis p-value = 3.93e-6
# MKI67 p-value = 0.017247
# AIC = 99.573

## ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_MKI67, glm_age_and_metastasis_at_diagnosis)
# p-value = 0.02101816
}
}

# Age, Metastasis at Diagnosis and ARNTL ####
{
## LM ####

{
### Model ####

lm_age_metastasis_at_diagnosis_and_ARNTL =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + ARNTL)

## ANOVA ####

anova_death_by_ARNTL_on_top_of_age_and_metastasis_at_diagnosis =
  anova(lm_age_and_metastasis_at_diagnosis,
        lm_age_metastasis_at_diagnosis_and_ARNTL)
# p-value = 0.5167
}

## GLM ####

{
### Model ####
glm_age_metastasis_at_diagnosis_and_ARNTL =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + ARNTL,
      family = "binomial") %>%
  summary
# age p-value = 0.000873
# Metastasis.At.Diagnosis p-value = 0.001794
# No.Metastasis.At.Diagnosis p-value = 1.92e-5
# ARNTL p-value = 0.431593
# AIC = 104.31

## ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_ARNTL, glm_age_and_metastasis_at_diagnosis)
# p-value = 0.4428085

}
}
# Age, Metastasis at Diagnosis and Metastasis within 5 Years of Diagnosis ####
{
## LM ####

{
### Model ####

lm_age_metastasis_at_diagnosis_and_metastasis_within_5_years =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + New.Tumor.Event.Within.5.Years)

## ANOVA ####

anova_death_by_metastasis_within_5_years_on_top_of_age_and_metastasis_at_diagnosis =
  anova(lm_age_and_metastasis_at_diagnosis,
        lm_age_metastasis_at_diagnosis_and_metastasis_within_5_years)
# p-value = 0.005538
}

## GLM ####

{
### Model ####

glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + New.Tumor.Event.Within.5.Years,
      family  = "binomial") %>% 
  summary

## ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years, glm_age_and_metastasis_at_diagnosis)
# p-value = 6.168553e-20

}
}

# Age and Multiple Node Infiltration at Diagnosis ####
{
## LM ####

{
### Model ####

lm_age_and_multiple_node_infiltration_at_diagnosis =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Node.Infiltration.At.Diagnosis)

## ANOVA ####

anova_death_by_multiple_node_infiltration_at_diagnosis_on_top_of_age =
  anova(lm_age,
        lm_age_and_multiple_node_infiltration_at_diagnosis)
# p-value = 0.2561

}

## GLM ####


{
### Model ####

glm_age_and_multiple_node_infiltration_at_diagnosis =
  glm(data    = sample_fit,
      formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Node.Infiltration.At.Diagnosis,
      family  = "binomial") %>%
  summary
# age p-value = 0.00263
# Node.Infiltration.At.Diagnosis p-value = 0.20507
# No.Node.Infiltration.At.Diagnosis p-value = 1.37e-05
# AIC = 109.49

## ChiSq ####

get_nested_glm_p_value(glm_age_and_multiple_node_infiltration_at_diagnosis, glm_age)
# p-value = 0.2270206
}
}
}

# 3 Tier Magnitude ####
{
## Contingency Table of Death within 5 Years of Diagnosis and 3-Tier Magnitude Group ####

table_died_within_5_years_of_diagnosis_by_mag_group = sample_fit %>% 
  dplyr::select(Died.Within.5.Years.Of.Diagnosis, Magnitude_Group) %>% 
  table

{
### Save Contingency Table to CSV ####

# save_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Figure3/Tumor_Sample_Magnitude/files/Adverse_Event_Proportion_v2"
# save_name = paste(save_path, "Death_within_5_Years_by_Magnitude_Group_3_Tier.csv", sep = "/")
# 
# readr::write_csv(table_died_within_5_years_of_diagnosis_by_mag_group %>% as.data.frame.matrix %>% rownames_to_column(var = "Death.within.5.Years"),
#                  file = save_name)

## Fisher Exact Test ####

fisher.test(table_died_within_5_years_of_diagnosis_by_mag_group) # p-value = 0.05219
}

# Nested Models ####
{
## Age and Magnitude Group ####
{
### LM ####

{
#### Model ####
  
lm_age_and_magnitude_group = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Magnitude_Group)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_age = 
  anova(lm_age, 
        lm_age_and_magnitude_group)
# p-value = 0.04049

}
## GLM ####

{
### Model ####
glm_age_and_magnitude_group = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Magnitude_Group,
      family  = "binomial") %>%
  summary
# age p-value = 0.00363
# Low.Magnitude.Group p-value = 2.58e-05
# Med.Magnitude.Group p-value = 0.12092
# High.Magnitude.Group p-value = 0.47473
# AIC = 106.41
  
## ChiSq ####

get_nested_glm_p_value(glm_age_and_magnitude_group, glm_age)
# p-value = 0.03807563

}
}

# Age, MKI67 and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_MKI67_and_magnitude_group =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + MKI67 + Magnitude_Group)

### ANOVA ####
anova_death_by_magnitude_group_on_top_of_age_and_MKI67 =
  anova(lm_age_and_MKI67,
        lm_age_MKI67_and_magnitude_group)
# p-value = 0.03834
}

## GLM ####

{
#### Model ####

glm_age_MKI67_and_magnitude_group =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + MKI67 + Magnitude_Group,
      family = "binomial") %>%
  summary
# age p-value = 0.000687
# Low.Magnitude.Group p-value = 9.3e-6
# Med.Magnitude.Group p-value = 0.91432
# High.Magnitude.Group p-value = 0.520988
# MKI67 p-value = 0.006396
# AIC = 101.32

### ChiSq ####
get_nested_glm_p_value(glm_age_MKI67_and_magnitude_group, glm_age_and_MKI67)
# p-value = 0.02816565

}
}
# Age, Adverse Event at Diagnosis and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_adverse_event_at_diagnosis_and_magnitude_group =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Adverse.Event.At.Diagnosis + Magnitude_Group)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_adverse_event_at_diagnosis_and_age = 
  anova(lm_age_and_adverse_event_at_diagnosis,
        lm_age_adverse_event_at_diagnosis_and_magnitude_group)
# p-value = 0.04359

}
## GLM ####

{
#### Model ####

glm_age_adverse_event_at_diagnosis_and_magnitude_group =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Adverse.Event.At.Diagnosis + Magnitude_Group,
      family  = "binomial") %>% 
  summary
# age p-value = 0.00268
# Adverse.Event.At.Diagnosis p-value = 0.07416
# No.Adverse.Event.At.Diagnosis.Low.Magnitude.Group p-value = 1.71e-05
# Med.Magnitude.Group p-value = 0.18603
# High.Magnitude.Group p-value = 0.32237
# AIC = 105.45

### ChiSq ####

get_nested_glm_p_value(glm_age_adverse_event_at_diagnosis_and_magnitude_group, glm_age_and_adverse_event_at_diagnosis)
# p-value = 0.04117713

}
}

# Age, Metastasis at Diagnosis and Magnitude Group ####
{
### LM ####

{
#### Model ####
  
lm_age_metastasis_at_diagnosis_and_magnitude_group =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + Magnitude_Group)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_age_and_metastasis_at_diagnosis =
  anova(lm_age_and_metastasis_at_diagnosis,
        lm_age_metastasis_at_diagnosis_and_magnitude_group)
# p-value = 0.02191

}

## GLM ####

{
#### Model ####

glm_age_metastasis_at_diagnosis_and_magnitude_group = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + Magnitude_Group,
      family  = "binomial") %>%
  summary
# age p-value = 0.00118
# Metastasis.At.Diagnosis p-value = 0.00136
# No.Metastasis.At.Diagnosis.Low.Magnitude.Group p-value = 1.69e-05
# Med.Magnitude.Group p-value = 0.07957
# High.Magnitude.Group p-value = 0.50090
# AIC = 98.922

### ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_magnitude_group, glm_age_and_metastasis_at_diagnosis)
# p-value = 0.0185396

}
}

# Age, Metastasis at Diagnosis, MKI67 and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_metastasis_at_diagnosis_MKI67_and_magnitude_group =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + MKI67 + Magnitude_Group)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_age_metastasis_at_diagnosis_and_MKI67 =
  anova(lm_age_metastasis_at_diagnosis_and_MKI67,
        lm_age_metastasis_at_diagnosis_MKI67_and_magnitude_group)
# p-value = 0.02084
}

## GLM ####

{
#### Model ####

glm_age_metastasis_at_diagnosis_MKI67_and_magnitude_group =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + MKI67 + Magnitude_Group,
      family = "binomial") %>%
  summary
# age p-value = 0.000321
# Metastasis.At.Diagnosis p-value = 0.003127
# No.Metastasis.At.Diagnosis.Low.Magnitude.Group p-value = 8.47e-6
# Med.Magnitude.Group p-value = 0.071073
# High.Magnitude.Group p-value = 0.498008
# MKI67 p-value = 0.011515
# AIC = 94.777

### ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_MKI67_and_magnitude_group, glm_age_metastasis_at_diagnosis_and_MKI67)
# p-value = 0.01230503
}
}
# Age, Metastasis at Diagnosis, ARNTL and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_metastasis_at_diagnosis_ARNTL_and_magnitude_group =
  lm(data = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + ARNTL + Magnitude_Group)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_age_metastasis_at_diagnosis_and_ARNTL =
  anova(lm_age_metastasis_at_diagnosis_and_ARNTL,
        lm_age_metastasis_at_diagnosis_ARNTL_and_magnitude_group)
# p-value = 0.02638
}

## GLM ####

{
#### Model ####

glm_age_metastasis_at_diagnosis_ARNTL_and_magnitude_group =
  glm(data = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + ARNTL + Magnitude_Group,
      family = "binomial") %>%
  summary
# age p-value = 0.00115
# Metastasis.At.Diagnosis p-value = 0.00125
# No.Metastasis.At.Diagnosis.Low.Magnitude.Group p-value = 4.17e-5
# Med.Magnitude.Group p-value = 0.07502
# High.Magnitude.Group p-value = 0.57558
# AIC = 100.61

### ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_ARNTL_and_magnitude_group, glm_age_metastasis_at_diagnosis_and_ARNTL)
# p-value = 0.02129958
}
}
# Age, Metastasis at Diagnosis, Metastasis within 5 Years of Diagnosis, and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_metastasis_at_diagnosis_and_within_5_years_and_magnitude_group =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + New.Tumor.Event.Within.5.Years + Magnitude_Group)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_age_metastasis_at_diagnosis_and_within_5_years =
  anova(lm_age_metastasis_at_diagnosis_and_metastasis_within_5_years,
        lm_age_metastasis_at_diagnosis_and_within_5_years_and_magnitude_group)
# p-value = 0.009731

}

## GLM ####

{
#### Model ####

glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years_and_magnitude_group =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + New.Tumor.Event.Within.5.Years + Magnitude_Group,
      family  = "binomial") %>%
  summary
# age p-value = 0.002690
# Metastasis.At.Diagnosis p-value = 0.000945
# New.Tumor.Event.Within.5.Years p-value = 0.004479
# Med.Magnitude.Group p-value = 0.074689
# High.Magnitude.Group p-value = 0.173679
# No.Metastasis.At.Diagnosis.No.New.Tumor.Event.Within.5.Years.Low.Magnitude.Group p-value = 2.29e-05
# AIC = 93.404

### ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years_and_magnitude_group, glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years)
# p-value = 0.003117404

}
}
}
}

# 2 Tier Magnitude ####
{
## Contingency Table of Death within 5 Years of Diagnosis and 2-Tier Magnitude Group ####
  
table_died_within_5_years_of_diagnosis_by_mag_group_low_med_merged = sample_fit %>% 
  dplyr::select(Died.Within.5.Years.Of.Diagnosis, Mag_Group_Low_Med_merged) %>% 
  table
  
{
### Save Contingency Table to CSV ####

# save_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Figure3/Tumor_Sample_Magnitude/files/Adverse_Event_Proportion_v2"
# save_name = paste(save_path, "Death_within_5_Years_by_Magnitude_Group_2_Tier.csv", sep = "/")
# 
# readr::write_csv(table_died_within_5_years_of_diagnosis_by_mag_group_low_med_merged %>% 
#                    as.data.frame.matrix %>% 
#                    'colnames<-'(c("High", "Low.or.Med")) %>%
#                    rownames_to_column(var = "Death.within.5.Years"),
#                  file = save_name)

## Fisher Exact Test ####

fisher.test(table_died_within_5_years_of_diagnosis_by_mag_group_low_med_merged) # p-value = 0.03107
}

# Nested Models ####
{
## Age and Magnitude Group ####
{
### LM ####
  
{
#### Model ####

lm_age_and_magnitude_group_low_med_merged = 
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Mag_Group_Low_Med_merged)

### ANOVA ####

anova_death_by_magnitude_group_low_med_merged_on_top_of_age = 
  anova(lm_age, 
        lm_age_and_magnitude_group_low_med_merged)
# p-value = 0.03572

}

## GLM ####

{
### Model ####

glm_age_and_magnitude_group_low_med_merged = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Mag_Group_Low_Med_merged,
      family  = "binomial") %>%
  summary
# age p-value = 0.005739
# Low.or.Med.Magnitude.Group p-value = 0.051718
# High.Magnitude.Group p-value = 0.000253
# AIC = 107.12

## ChiSq ####

get_nested_glm_p_value(glm_age_and_magnitude_group_low_med_merged, glm_age)
# p-value = 0.05021651

}
}
  
# Age, Adverse Event at Diagnosis and Magnitude Group ####
{
### LM ####

{
#### Model ####
      
lm_age_adverse_event_at_diagnosis_and_magnitude_group_low_med_merged =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Adverse.Event.At.Diagnosis + Mag_Group_Low_Med_merged)

### ANOVA ####

anova_death_by_magnitude_group_low_med_merged_on_top_of_adverse_event_at_diagnosis_and_age = 
  anova(lm_age_and_adverse_event_at_diagnosis,
        lm_age_adverse_event_at_diagnosis_and_magnitude_group_low_med_merged)
# p-value = 0.02973
  
}
## GLM ####

{
### Model ####
      
glm_age_adverse_event_at_diagnosis_and_magnitude_group_low_med_merged =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Adverse.Event.At.Diagnosis + Mag_Group_Low_Med_merged,
      family  = "binomial") %>% 
  summary
# age p-value = 0.003449
# Adverse.Event.At.Diagnosis p-value = 0.043346
# No.Adverse.Event.At.Diagnosis.High.Magnitude.Group p-value = 0.000127
# Low.or.Med.Magnitude.Group p-value = 0.036923
# AIC = 105.38

## ChiSq ####

get_nested_glm_p_value(glm_age_adverse_event_at_diagnosis_and_magnitude_group_low_med_merged, glm_age_and_adverse_event_at_diagnosis)
# p-value = 0.0348346
      
}
}
  
# Age, Metastasis at Diagnosis and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_metastasis_at_diagnosis_and_magnitude_group_low_med_merged =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + Mag_Group_Low_Med_merged)

### ANOVA ####

anova_death_by_magnitude_group_on_top_of_age_and_metastasis_at_diagnosis =
  anova(lm_age_and_metastasis_at_diagnosis,
        lm_age_metastasis_at_diagnosis_and_magnitude_group_low_med_merged)
# p-value = 0.02678

}

## GLM ####

{
### Model ####

glm_age_metastasis_at_diagnosis_and_magnitude_group_low_med_merged = 
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + Mag_Group_Low_Med_merged,
      family  = "binomial") %>%
  summary
# age p-value = 0.001876
# Metastasis.At.Diagnosis p-value = 0.001231
# No.Metastasis.At.Diagnosis.High.Magnitude.Group p-value = 0.000116
# Low.or.Med.Magnitude.Group p-value = 0.038374
# AIC = 100.46

## ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_magnitude_group_low_med_merged, glm_age_and_metastasis_at_diagnosis)
# p-value = 0.03520762

}
}
  
# Age, Metastasis at Diagnosis, Metastasis within 5 Years of Diagnosis, and Magnitude Group ####
{
### LM ####

{
#### Model ####

lm_age_metastasis_at_diagnosis_and_within_5_years_and_magnitude_group_low_med_merged =
  lm(data    = sample_fit,
     formula = as.numeric(Died.Within.5.Years.Of.Diagnosis) ~ age + Metastasis.At.Diagnosis + New.Tumor.Event.Within.5.Years + Mag_Group_Low_Med_merged)

### ANOVA ####

anova_death_by_magnitude_group_low_med_merged_on_top_of_age_metastasis_at_diagnosis_and_within_5_years =
  anova(lm_age_metastasis_at_diagnosis_and_metastasis_within_5_years,
        lm_age_metastasis_at_diagnosis_and_within_5_years_and_magnitude_group_low_med_merged)
# p-value = 0.009568

}

## GLM ####

{
### Model ####

glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years_and_magnitude_group_low_med_merged =
  glm(data    = sample_fit,
      formula = Died.Within.5.Years.Of.Diagnosis ~ age + Metastasis.At.Diagnosis + New.Tumor.Event.Within.5.Years + Mag_Group_Low_Med_merged,
      family  = "binomial") %>%
  summary
# age p-value = 0.004512
# Metastasis.At.Diagnosis p-value = 0.000705
# New.Tumor.Event.Within.5.Years p-value = 0.004423
# Low.or.Med.Magnitude.Group p-value = 0.009099
# High.Magnitude.Group p-value = 0.173679
# No.Metastasis.At.Diagnosis.No.New.Tumor.Event.Within.5.Years.High.Magnitude.Group p-value = 0.000238
# AIC = 95.066

## ChiSq ####

get_nested_glm_p_value(glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years_and_magnitude_group_low_med_merged, glm_age_metastasis_at_diagnosis_and_metastasis_within_5_years)
# p-value = 0.00499947

}
}
}
}

# Plots ####
{
## 2 Tier Magnitude Group ####
{
### Plot ####
death_within_5_years_by_magnitude_group_low_med_merged_bar_plot = sample_fit %>%
  group_by(Mag_Group_Low_Med_merged) %>%
  summarise(sum(Died.Within.5.Years.Of.Diagnosis)/length(Died.Within.5.Years.Of.Diagnosis)) %>%
  as.data.frame %>%
  'colnames<-'(c("Magnitude.Group", "Proportion")) %>%
  
  ggplot() +
  geom_col(mapping = aes(y    = Proportion,
                         x    = factor(Magnitude.Group, levels = c("TRUE", "FALSE")),
                         fill = Magnitude.Group)) +
  scale_x_discrete(breaks = c("TRUE", "FALSE"),
                   labels = c("Low.or.Med", "High"),
                   name   = "Magnitude Group") +
  scale_fill_manual(guide = NULL,
                    values = c("TRUE" = "navy", "FALSE" = "red")) +
  scale_y_continuous(name = "Percent Death\nwithin 5 Years",
                     breaks = c(0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3),
                     labels = c("0.0%", "2.5%", "5.0%", "7.5%", "10.0%", "12.5%", "15.0%", "17.5%", "20.0%", "22.5%", "25.0%", "27.5%", "30.0%"),
                     limits = c(0, 0.3)) +
  quick_theme()

## Save Plot ####

# save_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Figure3/Tumor_Sample_Magnitude/image_files/Adverse_Event_Proportion_v2"
# save_name = paste(save_path, "Percent_Death_within_5_Years_by_Magnitude_Group_2_Tier.pdf", sep = "/")
# 
# ggsave(plot     = death_within_5_years_by_magnitude_group_low_med_merged_bar_plot,
#        filename = save_name,
#        height   = 8,
#        width    = 5,
#        unit     = "in")

}

# 3 Tier Magnitude Group ####
{
## Plot 1 ####
death_within_5_years_by_magnitude_group_bar_plot = sample_fit %>%
  group_by(Magnitude_Group) %>%
  summarise(sum(Died.Within.5.Years.Of.Diagnosis)/length(Died.Within.5.Years.Of.Diagnosis)) %>%
  as.data.frame %>%
  'colnames<-'(c("Magnitude.Group", "Proportion")) %>%
  
  ggplot() +
  geom_col(mapping = aes(y    = Proportion,
                         x    = Magnitude.Group,
                         fill = Magnitude.Group)) +
  scale_x_discrete(breaks = c("Low", "Med", "High"),
                   labels = c("Low", "Medium", "High"),
                   name   = "Magnitude Group") +
  scale_fill_manual(guide = NULL,
                    values = c("Low" = "navy", "Med" = "turquoise", "High" = "red")) +
  scale_y_continuous(name = "Percent Death\nwithin 5 Years",
                     breaks = c(0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3),
                     labels = c("0.0%", "2.5%", "5.0%", "7.5%", "10.0%", "12.5%", "15.0%", "17.5%", "20.0%", "22.5%", "25.0%", "27.5%", "30.0%"),
                     limits = c(0, 0.3)) +
  quick_theme()

# Save Plot 1 ####

# save_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Figure3/Tumor_Sample_Magnitude/image_files/Adverse_Event_Proportion_v2"
# save_name = paste(save_path, "Percent_Death_within_5_Years_by_Magnitude_Group_3_Tier.pdf", sep = "/")
# 
# ggsave(plot     = death_within_5_years_by_magnitude_group_bar_plot,
#        filename = save_name,
#        height   = 8,
#        width    = 5,
#        unit     = "in")

# Plot 2 ####

death_within_5_years_by_magnitude_group_bar_plot_share = sample_fit %>%
  group_by(Magnitude_Group) %>%
  summarise(sum(Died.Within.5.Years.Of.Diagnosis)/length(Died.Within.5.Years.Of.Diagnosis)) %>%
  as.data.frame %>%
  'colnames<-'(c("Magnitude.Group", "Proportion")) %>%
  
  ggplot() +
  geom_col(mapping = aes(y    = Proportion,
                         x    = Magnitude.Group,
                         fill = Magnitude.Group)) +
  scale_x_discrete(breaks = c("Low", "Med", "High"),
                   labels = c("Low", "Medium", "High"),
                   name   = "Magnitude Group") +
  scale_fill_manual(guide = NULL,
                    values = c("Low" = "navy", "Med" = "turquoise", "High" = "red")) +
  scale_y_continuous(name = "Percent Death\nwithin 5 Years",
                     breaks = c(0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15),
                     labels = c("0.0%", "2.5%", "5.0%", "7.5%", "10.0%", "12.5%", "15.0%"),
                     limits = c(0, 0.15)) +
  quick_theme()

# Save Plot 2 ####

# save_path = "/Users/janhammarlund/Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 2/PaperFigures/Figure3/Tumor_Sample_Magnitude/image_files/Adverse_Event_Proportion_v2"
# save_name = paste(save_path, "Percent_Death_within_5_Years_by_Magnitude_Group_3_Tier_share.pdf", sep = "/")
# 
# ggsave(plot     = death_within_5_years_by_magnitude_group_bar_plot_share,
#        filename = save_name,
#        height   = 8,
#        width    = 5,
#        unit     = "in")

}
}
