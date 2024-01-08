using DataFrames, Distributed, CSV

base_path = joinpath(homedir(), "Library/CloudStorage/Box-Box/PhD_Jan_Hammarlund/Specific Aim 1") # define the base path in which the data file, the seed gene list, and the sample collection time file are located
data_path = joinpath(base_path, "Data") # path for data folder in base folder
seed_path = joinpath(base_path, "Data") # path for data folder in base folder
output_path_warmup = joinpath(base_path, "Individual_Training", "output_warmup") # warmup output folder in base folder
output_path = joinpath(base_path, "Individual_Training") # real training run output folder in base folder

file_name = "BA11_seed_data.csv"
file_df = CSV.read(joinpath(data_path, file_name), DataFrame)
file_df[24,1] = "CIART"
full_file_name = "Annotated_Unlogged_BA11Data.csv"
full_file_df = CSV.read(joinpath(data_path, full_file_name), DataFrame)
full_file_clean_matrix = Matrix(full_file_df[3:end,vcat(2,collect(4:205))])
full_file_exp_matrix = parse.(Float64, string.(full_file_clean_matrix[:,2:end]))
full_file_df_clean = hcat(DataFrame(Gene_Symbol=Array{String,1}(full_file_clean_matrix[:,1])), DataFrame(full_file_exp_matrix, names(full_file_df)[4:205]))

seed = CSV.read(joinpath(seed_path, "HUC_original.csv"), DataFrame).x1

# collection_time_df = CSV.read(joinpath(seed_path, "BA11_collection_times.csv"), DataFrame) # load collection time file in data folder
collection_time_df = CSV.read(joinpath(seed_path, "BA11_collection_times.csv"), DataFrame) # load collection time file in data folder

# sample_ids_with_collection_times = names(collection_time_df) # get sample ids from collection time file
# sample_ids_with_collection_times_full = vcat(sample_ids_with_collection_times, map(x -> join([x, "_1"]), sample_ids_with_collection_times))
sample_ids_with_collection_times_full = names(collection_time_df)
# sample_collection_times = collect(collection_time_df[1, :]) # get sample collection times from collection time file
# sample_collection_times_full = repeat(sample_collection_times, 2)
sample_collection_times_full = collect(collection_time_df[1,:])
# CSV.write(joinpath(seed_path, "Double_BA11_collection_times.csv"), DataFrame(hcat(sample_collection_times_full...), sample_ids_with_collection_times_full))

# make changes to training parameters, if required. Below are the defaults for the current version of cyclops.
training_parameters = Dict(:regex_cont => r".*_C",			# What is the regex match for continuous covariates in the data file
:regex_disc => r".*_D",							# What is the regex match for discontinuous covariates in the data file

:blunt_percent => 0.975, 						# What is the percentile cutoff below (lower) and above (upper) which values are capped

:seed_min_CV => 0.14, 							# The minimum coefficient of variation a gene of interest may have to be included in eigen gene transformation
:seed_max_CV => 0.9, 							# The maximum coefficient of a variation a gene of interest may have to be included in eigen gene transformation
:seed_mth_Gene => 10000, 						# The minimum mean a gene of interest may have to be included in eigen gene transformation

:norm_gene_level => true, 						# Does mean normalization occur at the seed gene level
:norm_disc => false, 							# Does batch mean normalization occur at the seed gene level
:norm_disc_cov => 1, 							# Which discontinuous covariate is used to mean normalize seed level data

:eigen_reg => true, 							# Does regression again a covariate occur at the eigen gene level
:eigen_reg_disc_cov => 1, 						# Which discontinous covariate is used for regression
#~~~New Addition~~~#
:eigen_reg_exclude => false,						# Are eigen genes with r squared greater than cutoff removed from final eigen data output
#~~~~~~~~~~~~~~~~~~#
:eigen_reg_r_squared_cutoff => 0.6,				# This cutoff is used to determine whether an eigen gene is excluded from final eigen data used for training
:eigen_reg_remove_correct => false,				# Is the first eigen gene removed (true --> default) or it's contributed variance of the first eigne gene corrected by batch regression (false)

:eigen_first_var => false, 						# Is a captured variance cutoff on the first eigen gene used
:eigen_first_var_cutoff => 0.85, 				# Cutoff used on captured variance of first eigen gene

:eigen_total_var => 0.85, 						# Minimum amount of variance required to be captured by included dimensions of eigen gene data
:eigen_contr_var => 0.03, 						# Minimum amount of variance required to be captured by a single dimension of eigen gene data
:eigen_var_override => false,					# Is the minimum amount of contributed variance ignored
:eigen_max => 30, 								# Maximum number of dimensions allowed to be kept in eigen gene data

:out_covariates => true, 						# Are covariates included in eigen gene data
:out_use_disc_cov => true,						# Are discontinuous covariates included in eigen gene data
:out_all_disc_cov => true, 						# Are all discontinuous covariates included if included in eigen gene data
:out_disc_cov => 1,								# Which discontinuous covariates are included at the bottom of the eigen gene data, if not all discontinuous covariates
:out_use_cont_cov => false,						# Are continuous covariates included in eigen data
:out_all_cont_cov => true,						# Are all continuous covariates included in eigen gene data
:out_use_norm_cont_cov => false,				# Are continuous covariates Normalized
:out_all_norm_cont_cov => true,					# Are all continuous covariates normalized
:out_cont_cov => 1,								# Which continuous covariates are included at the bottom of the eigen gene data, if not all continuous covariates, or which continuous covariates are normalized if not all
:out_norm_cont_cov => 1,						# Which continuous covariates are normalized if not all continuous covariates are included, and only specific ones are included

:init_scale_change => true,						# Are scales changed
:init_scale_1 => false,							# Are all scales initialized such that the model sees them all as having scale 1
                                                # Or they'll be initilized halfway between 1 and their regression estimate.

:train_n_models => 80, 							# How many models are being trained
:train_μA => 0.001, 							# Learning rate of ADAM optimizer
:train_β => (0.9, 0.999), 						# β parameter for ADAM optimizer
#:train_optimizer => ADAM(0.0001, (0.9, 0.999)),	# optimizer to be used in training of model
:train_min_steps => 1500, 						# Minimum number of training steps per model
:train_max_steps => 2050, 						# Maximum number of training steps per model
:train_μA_scale_lim => 1000, 					# Factor used to divide learning rate to establish smallest the learning rate may shrink to
:train_circular => false,						# Train symmetrically
:train_collection_times => false,						# Train using known times
:train_collection_time_balance => 0.5,					# How is the true time loss rescaled
:train_sample_id => sample_ids_with_collection_times_full,
:train_sample_phase => sample_collection_times_full,

:cosine_shift_iterations => 192,				# How many different shifts are tried to find the ideal shift
:cosine_covariate_offset => true,				# Are offsets calculated by covariates

:align_p_cutoff => 0.1,						    # When aligning the acrophases, what genes are included according to the specified p-cutoff
:align_base => "radians",						# What is the base of the list (:align_acrophases or :align_phases)? "radians" or "hours"
:align_disc => false,							# Is a discontinuous covariate used to align (true or false)
:align_disc_cov => 1,							# Which discontinuous covariate is used to choose samples to separately align (is an integer)
:align_other_covariates => false,				# Are other covariates included
:align_batch_only => false,
# :align_genes => Array{String, 1},				# A string array of genes used to align CYCLOPS fit output. Goes together with :align_acrophases
# :align_acrophases => Array{<: Number, 1}, 	# A number array of acrophases for each gene used to align CYCLOPS fit output. Goes together with :align_genes
:align_samples => sample_ids_with_collection_times_full,
:align_phases => sample_collection_times_full,

:X_Val_k => 10,									# How many folds used in cross validation.
:X_Val_omit_size => 0.1,						# What is the fraction of samples left out per fold

:plot_use_o_cov => true,
:plot_correct_batches => false,
:plot_disc => false,
:plot_disc_cov => 1,
:plot_separate => false,
:plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
:plot_only_color => true,
:plot_p_cutoff => 0.05)

Distributed.addprocs(length(Sys.cpu_info()))
@everywhere include("/Users/janhammarlund/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Jan’s MacBook Pro/GitCYCLOPS.nosync/CYCLOPS/CYCLOPS.jl")

full_file_df_clean[CYCLOPS.findXinY(["C1orf51"], full_file_df_clean[:,1])[1],1] = "CIART"

training_parameters[:align_genes] = CYCLOPS.human_homologue_gene_symbol[CYCLOPS.human_homologue_gene_symbol.!="RORC"]
training_parameters[:align_acrophases] = CYCLOPS.mouse_acrophases[CYCLOPS.human_homologue_gene_symbol.!="RORC"]

eigendata, modeloutputs, metric_correlations, bestmodel, parameters = CYCLOPS.Fit(full_file_df_clean, seed, training_parameters)
# CYCLOPS.Align(file_df, modeloutputs, metric_correlations, bestmodel, parameters, output_path)
CYCLOPS.Align(full_file_df_clean, modeloutputs, metric_correlations, bestmodel, parameters, joinpath(output_path, "Original_Training_Run_Parameters_Rerun_05_05_2023"))

