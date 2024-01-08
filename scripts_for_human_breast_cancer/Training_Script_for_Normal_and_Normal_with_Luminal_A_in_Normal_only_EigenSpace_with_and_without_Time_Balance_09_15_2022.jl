using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats, PyPlot, Distributed, Random, CSV, Revise, Distributions, Dates, MultipleTesting

base_path = joinpath(homedir(), "Library", "CloudStorage", "Box-Box", "Share_Jan_and_Ron", "GTEx_TCGA_UK", "TNT_Training") # define the base path in which the data file, the seed gene list, and the sample collection time file are located
data_path = joinpath(base_path, "processed_data") # path for data folder in base folder
seed_path = joinpath(base_path, "unprocessed_data") # path for data folder in base folder
output_path_warmup = joinpath(base_path, "output_warmup") # warmup output folder in base folder
output_path = joinpath(base_path, "training_results") # real training run output folder in base folder

raw_TPM = CSV.read(joinpath(data_path, "gtex_tcga_uk_non_tumor_and_tumor_data-batches_with_more_than_5_samples_12_01_21.csv"), DataFrame) # load data file in data folder
NT_raw_TPM_F = raw_TPM[:,vcat(true, collect(raw_TPM[1,2:end]).=="NonTumor")]
NT_raw_TPM = NT_raw_TPM_F[2:end,:]
NT_tcga_TPM = NT_raw_TPM[:, map(x -> !isa(match(r"GTEX", x), RegexMatch), names(NT_raw_TPM))]

LumA_raw_TPM_F = raw_TPM[:,vcat(true, collect(raw_TPM[1,2:end]) .=="LumA")]
LumA_raw_TPM = LumA_raw_TPM_F[2:end,:]

LumB_raw_TPM_F = raw_TPM[:,vcat(true, collect(raw_TPM[1,2:end]) .=="LumB")]
LumB_raw_TPM = LumB_raw_TPM_F[2:end,:]

Basal_raw_TPM_F = raw_TPM[:,vcat(true, collect(raw_TPM[1,2:end]) .=="Basal")]
Basal_raw_TPM = Basal_raw_TPM_F[2:end,:]

Her2_raw_TPM_F = raw_TPM[:,vcat(true, collect(raw_TPM[1,2:end]) .=="Her2")]
Her2_raw_TPM = Her2_raw_TPM_F[2:end,:]

NT_LumA_raw_TPM_F = hcat(NT_raw_TPM_F, LumA_raw_TPM_F[:, 2:end])
NT_LumB_raw_TPM_F = hcat(NT_raw_TPM_F, LumB_raw_TPM_F[:, 2:end])
NT_Basal_raw_TPM_F = hcat(NT_raw_TPM_F, Basal_raw_TPM_F[:, 2:end])
NT_Basal_Her2_raw_TPM_F = hcat(NT_raw_TPM_F, Basal_raw_TPM_F[:, 2:end], Her2_raw_TPM_F[:, 2:end])

zhang_meng_bhq_0_01_rAmp = unique(CSV.read(joinpath(seed_path, "Zhang_Meng_bhq_0_01_rAmp_0_33.csv"), DataFrame)).Seed_Symbol

collection_time_df = CSV.read(joinpath(seed_path, "UK_non_tumor_collection_time.csv"), DataFrame) # load collection time file in data folder

sample_ids_with_collection_times = names(collection_time_df)
sample_collection_times = collect(collection_time_df[1, :]) # get sample collection times from collection time file

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
:eigen_contr_var => 0.05, 						# Minimum amount of variance required to be captured by a single dimension of eigen gene data
:eigen_var_override => true,					# Is the minimum amount of contributed variance ignored
:eigen_max => 5, 								# Maximum number of dimensions allowed to be kept in eigen gene data

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
# :train_circular => false,						# Train symmetrically
# :train_collection_times => true,						# Train using known times
# :train_collection_time_balance => 1.0,					# How is the true time loss rescaled
# :train_sample_id => sample_ids_with_collection_times,
# :train_sample_phase => sample_collection_times,

:cosine_shift_iterations => 192,				# How many different shifts are tried to find the ideal shift
:cosine_covariate_offset => true,				# Are offsets calculated by covariates

:align_p_cutoff => 0.05,						# When aligning the acrophases, what genes are included according to the specified p-cutoff
:align_base => "radians",						# What is the base of the list (:align_acrophases or :align_phases)? "radians" or "hours"
:align_disc => false,							# Is a discontinuous covariate used to align (true or false)
:align_disc_cov => 1,							# Which discontinuous covariate is used to choose samples to separately align (is an integer)
:align_other_covariates => false,				# Are other covariates included
:align_batch_only => false,
# :align_samples => sample_ids_with_collection_times,
# :align_phases => sample_collection_times,
# :align_genes => Array{String, 1},				# A string array of genes used to align CYCLOPS fit output. Goes together with :align_acrophases
# :align_acrophases => Array{<: Number, 1}, 	# A number array of acrophases for each gene used to align CYCLOPS fit output. Goes together with :align_genes

:X_Val_k => 10,									# How many folds used in cross validation.
:X_Val_omit_size => 0.1,						# What is the fraction of samples left out per fold

:plot_use_o_cov => true,
:plot_correct_batches => true,
:plot_disc => false,
:plot_disc_cov => 1,
:plot_separate => false,
:plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
:plot_only_color => true,
:plot_p_cutoff => 0.05)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# warmup parameters to run CYCLOPS for the first time in a new julia session (speeds up subsequent CYCLOPS runs)
training_parameter_warmup = Dict{Symbol,Any}(:train_min_steps => 2, :train_max_steps => 2, :train_n_models => length(Sys.cpu_info()), :eigen_reg_exclude => false, :eigen_reg_remove_correct => false, :eigen_contr_var => 0.03, :eigen_var_override => true)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
Distributed.addprocs(length(Sys.cpu_info()))
# git CYCLOPS version c401b99421a15814f1d8acc936bfd05d9fed21d2
@everywhere include(joinpath(homedir(), "Documents", "GitCYCLOPS.nosync", "CYCLOPS", "CYCLOPS.jl"))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# warmup
eigendata, modeloutputs, correlations, bestmodel, parameters = CYCLOPS.Fit(NT_raw_TPM, zhang_meng_bhq_0_01_rAmp, training_parameter_warmup)
CYCLOPS.Align(NT_raw_TPM, modeloutputs, correlations, bestmodel, parameters, output_path_warmup)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Training Run
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
training_parameters[:align_genes], training_parameters[:align_acrophases] = CYCLOPS.human_homologue_gene_symbol[CYCLOPS.human_homologue_gene_symbol .!= "RORC"], CYCLOPS.mouse_acrophases[CYCLOPS.human_homologue_gene_symbol .!= "RORC"]
for ii in 1:2
    if ii == 2
        training_parameters[:train_sample_id], training_parameters[:train_sample_phase], training_parameters[:train_collection_time_balance] = sample_ids_with_collection_times, sample_collection_times, 510/299
    end
    for jj in [NT_raw_TPM, NT_LumA_raw_TPM_F]
        eigen_data, model_outputs, metric_correlations, trained_model, trained_parameters = CYCLOPS.TransferFit_d1(NT_raw_TPM, jj, Vector{String}(zhang_meng_bhq_0_01_rAmp), training_parameters)
        CYCLOPS.Align(jj, model_outputs, metric_correlations, trained_model, trained_parameters, output_path)
    end
end

function make_dict(parameter_csv::DataFrame)
    output_dict = Dict{Symbol,Any}()
    for ii in 1:length(parameter_csv[:, :first])
        jj = parameter_csv[ii, :first]
        kk = parameter_csv[ii, :second]
        if ismissing(kk)
        elseif kk == "radians"
            output_dict[Symbol(jj)] = kk
        elseif typeof(Meta.parse(kk)) .!= Expr
            output_dict[Symbol(jj)] = eval(Meta.parse(kk))
        elseif (Meta.parse(kk).args[1] == :(Array{String,1})) | (Meta.parse(kk).args[1] == :(Array{Int64,2})) | (Meta.parse(kk).args[1] == :(Vector{String15}))
            output_dict[Symbol(jj)] = eval.(Meta.parse(kk).args[2:end])
        else
            output_dict[Symbol(jj)] = eval(Meta.parse(kk))
        end
    end
    return output_dict
end

function build_model(the_path::String)
    model_csv = CSV.read(the_path, DataFrame)
    model_S_OH = eval(Meta.parse(model_csv[2,2]))
    model_B = eval(Meta.parse(model_csv[6,2]))
    model_B_OH = eval(Meta.parse(model_csv[4,2]))
    model_L1_w = eval(Meta.parse(model_csv[3,2]))
    model_L1_b = eval(Meta.parse(model_csv[5,2]))
    model_L1 = CYCLOPS.DenseFunction(model_L1_w, model_L1_b)
    model_L2_w = eval(Meta.parse(model_csv[7,2]))
    model_L2_b = eval(Meta.parse(model_csv[8,2]))
    model_L2 = CYCLOPS.DenseFunction(model_L2_w, model_L2_b)
    model_o = parse(Int64, model_csv[1,2])
    return CYCLOPS.Covariates(model_S_OH, model_B, model_B_OH, model_L1, model_L2, model_o)
end

original_NTLA_Fit_path = joinpath(output_path, "NTLA_2022-09-15T14_35_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_7", "Fits", "Fit_Output_2022-09-15T14_35_00.csv")
original_NTLA_Fit = CSV.read(original_NTLA_Fit_path, DataFrame)
original_NTLA_Cosine_path = joinpath(output_path, "NTLA_2022-09-15T14_35_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_7", "Fits", "Genes_of_Interest_Aligned_Cosine_Fit_2022-09-15T14_35_00.csv")
original_NTLA_Cosine = CSV.read(original_NTLA_Cosine_path, DataFrame)
original_NTLA_parameter_path = joinpath(output_path, "NTLA_2022-09-15T14_35_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_7", "Parameters", "Trained_Parameter_Dictionary_2022-09-15T14_35_00.csv")
original_parameters_csv = CSV.read(original_NTLA_parameter_path, DataFrame)
original_NTLA_parameters = make_dict(CSV.read(original_NTLA_parameter_path, DataFrame))
original_NT_samples_fit = original_NTLA_Fit[original_NTLA_Fit[:,:Covariate_D] .== "NonTumor", :]
original_LumA_samples_Fit = original_NTLA_Fit[original_NTLA_Fit[:,:Covariate_D] .== "LumA", :]

training_parameters[:align_genes], training_parameters[:align_acrophases] = CYCLOPS.human_homologue_gene_symbol[CYCLOPS.human_homologue_gene_symbol .!= "RORC"], CYCLOPS.mouse_acrophases[CYCLOPS.human_homologue_gene_symbol .!= "RORC"]
training_parameters[:train_sample_id], training_parameters[:train_sample_phase], training_parameters[:train_collection_time_balance] = sample_ids_with_collection_times, sample_collection_times, 510/299
model_path_NTLA = joinpath(output_path, "NTLA_2022-09-15T14_35_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_7", "Models", "Trained_Model_2022-09-15T14_35_00.csv")
tumor_output_path_LumA = joinpath(output_path, "NTLA_2022-09-15T14_35_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_7", "Luminal_A")
model_NTLA = build_model(model_path_NTLA)
LumA_transform, LumA_metricDataframe, LumA_correlations, ~, LumA_ops = CYCLOPS.ReApplyFit_d1(model_NTLA, NT_raw_TPM, NT_LumA_raw_TPM_F, LumA_raw_TPM_F, zhang_meng_bhq_0_01_rAmp, training_parameters)
~, (tumor_plot_path, ~, ~, ~) = CYCLOPS.Align(NT_LumA_raw_TPM_F, LumA_raw_TPM_F, original_NTLA_Fit, LumA_metricDataframe, LumA_correlations, model_NTLA, original_NTLA_parameters, LumA_ops, tumor_output_path_LumA)

training_parameters[:align_samples], training_parameters[:align_phases] = sample_ids_with_collection_times, sample_collection_times
NT_transform_LumA, NT_metricDataframe_LumA, NT_correlations_LumA, ~, NT_ops_LumA = CYCLOPS.ReApplyFit_d1(model_NTLA, NT_raw_TPM, NT_LumA_raw_TPM_F, NT_raw_TPM_F, zhang_meng_bhq_0_01_rAmp, training_parameters)
NT_output_path_LumA = joinpath(output_path, "NTLA_2022-09-15T14_35_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9_train_collection_time_balance_1_7", "Non_Tumor")
~, (tumor_plot_path, ~, ~, ~) = CYCLOPS.Align(NT_LumA_raw_TPM_F, NT_raw_TPM_F, original_NTLA_Fit, NT_metricDataframe_LumA, NT_correlations_LumA, model_NTLA, original_NTLA_parameters, NT_ops_LumA, NT_output_path_LumA)



original_NTLA_Fit_path = joinpath(output_path, "NTLA_2022-09-15T13_50_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9", "Fits", "Fit_Output_2022-09-15T13_50_00.csv")
original_NTLA_Fit = CSV.read(original_NTLA_Fit_path, DataFrame)

original_NTLA_parameter_path = joinpath(output_path, "NTLA_2022-09-15T13_50_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9", "Parameters", "Trained_Parameter_Dictionary_2022-09-15T13_50_00.csv")
original_parameters_csv = CSV.read(original_NTLA_parameter_path, DataFrame)
original_NTLA_parameters = make_dict(CSV.read(original_NTLA_parameter_path, DataFrame))

training_parameters[:align_genes], training_parameters[:align_acrophases] = CYCLOPS.human_homologue_gene_symbol[CYCLOPS.human_homologue_gene_symbol .!= "RORC"], CYCLOPS.mouse_acrophases[CYCLOPS.human_homologue_gene_symbol .!= "RORC"]
training_parameters[:train_sample_id], training_parameters[:train_sample_phase] = sample_ids_with_collection_times, sample_collection_times
model_path_NTLA = joinpath(output_path, "NTLA_2022-09-15T13_50_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9", "Models", "Trained_Model_2022-09-15T13_50_00.csv")
tumor_output_path_LumA = joinpath(output_path, "NTLA_2022-09-15T13_50_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9", "Luminal_A")
model_NTLA = build_model(model_path_NTLA)
LumA_transform, LumA_metricDataframe, LumA_correlations, ~, LumA_ops = CYCLOPS.ReApplyFit_d1(model_NTLA, NT_raw_TPM, NT_LumA_raw_TPM_F, LumA_raw_TPM_F, zhang_meng_bhq_0_01_rAmp, training_parameters)
~, (tumor_plot_path, ~, ~, ~) = CYCLOPS.Align(NT_LumA_raw_TPM_F, LumA_raw_TPM_F, original_NTLA_Fit, LumA_metricDataframe, LumA_correlations, model_NTLA, original_NTLA_parameters, LumA_ops, tumor_output_path_LumA)

training_parameters[:align_samples], training_parameters[:align_phases] = sample_ids_with_collection_times, sample_collection_times
NT_transform_LumA, NT_metricDataframe_LumA, NT_correlations_LumA, ~, NT_ops_LumA = CYCLOPS.ReApplyFit_d1(model_NTLA, NT_raw_TPM, NT_LumA_raw_TPM_F, NT_raw_TPM_F, zhang_meng_bhq_0_01_rAmp, training_parameters)
NT_output_path_LumA = joinpath(output_path, "NTLA_2022-09-15T13_50_00_eigen_contr_var_0_05_eigen_var_override_true_plot_correct_batches_true_seed_max_CV_0_9", "Non_Tumor")
~, (tumor_plot_path, ~, ~, ~) = CYCLOPS.Align(NT_LumA_raw_TPM_F, NT_raw_TPM_F, original_NTLA_Fit, NT_metricDataframe_LumA, NT_correlations_LumA, model_NTLA, original_NTLA_parameters, NT_ops_LumA, NT_output_path_LumA)
