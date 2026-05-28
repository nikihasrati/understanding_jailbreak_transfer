#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(performance)
  library(broom.mixed)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(name, default = NULL) {
  idx <- match(name, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

data_dir <- arg_value("--data-dir", "outputs/paper/feature_datasets")
results_dir <- arg_value("--results-dir", "outputs/paper/r/features")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) stop(sprintf("No CSV files found in %s", data_dir))

base_predictors <- c("baseline_score", "suffix_push", "orthogonal_shift")
semantic_predictors <- c(base_predictors, "semantic_sim_model")

standardize_predictors <- function(df, predictors) {
  for (pred in intersect(predictors, names(df))) {
    df[[paste0(pred, "_std")]] <- as.numeric(scale(df[[pred]]))
  }
  df
}

random_effects <- function(df) {
  dimensionality <- unique(df$dimensionality)[[1]]
  if (dimensionality == "1d") {
    "(1 | prompt_index) + (1 | suffix_index)"
  } else {
    "(1 | prompt_index) + (1 | source_prompt_index / suffix_index)"
  }
}

fit_glmm <- function(df, fixed_part) {
  formula <- as.formula(paste("jailbreak_success ~", fixed_part, "+", random_effects(df)))
  glmer(
    formula,
    data = df,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
}

decorate <- function(tidy_df, model, df, file, fit_kind) {
  r2_vals <- r2(model)
  tidy_df %>%
    mutate(
      file = basename(file),
      fit_kind = fit_kind,
      model_alias = unique(df$model_alias)[[1]],
      dimensionality = unique(df$dimensionality)[[1]],
      n = nrow(df),
      r2_marginal = r2_vals$R2_marginal,
      r2_conditional = r2_vals$R2_conditional
    )
}

fit_file <- function(path) {
  df <- read_csv(path, show_col_types = FALSE) %>%
    filter(prompt_index != source_prompt_index) %>%
    mutate(
      prompt_index = factor(prompt_index),
      suffix_index = factor(suffix_index),
      source_prompt_index = factor(source_prompt_index),
      jailbreak_success = as.integer(jailbreak_success)
    )
  df <- standardize_predictors(df, semantic_predictors)

  single <- bind_rows(lapply(base_predictors, function(pred) {
    model <- fit_glmm(df, paste0(pred, "_std"))
    decorate(tidy(model, effects = "fixed"), model, df, path, paste0("single_", pred))
  }))

  joint_fixed <- paste(
    "baseline_score_std + suffix_push_std + orthogonal_shift_std +",
    "baseline_score_std:suffix_push_std +",
    "baseline_score_std:orthogonal_shift_std +",
    "suffix_push_std:orthogonal_shift_std"
  )
  joint_model <- fit_glmm(df, joint_fixed)
  joint <- decorate(tidy(joint_model, effects = "fixed"), joint_model, df, path, "joint")

  semantic <- NULL
  if ("semantic_sim_model_std" %in% names(df)) {
    sem_fixed <- paste(
      "semantic_sim_model_std + baseline_score_std + suffix_push_std + orthogonal_shift_std +",
      "baseline_score_std:suffix_push_std +",
      "baseline_score_std:orthogonal_shift_std +",
      "suffix_push_std:orthogonal_shift_std +",
      "semantic_sim_model_std:baseline_score_std +",
      "semantic_sim_model_std:suffix_push_std +",
      "semantic_sim_model_std:orthogonal_shift_std"
    )
    sem_model <- fit_glmm(df, sem_fixed)
    semantic <- decorate(tidy(sem_model, effects = "fixed"), sem_model, df, path, "joint_with_semantic")
  }

  list(single = single, joint = joint, semantic = semantic)
}

fits <- lapply(files, fit_file)
single_rows <- bind_rows(lapply(fits, `[[`, "single"))
joint_rows <- bind_rows(lapply(fits, `[[`, "joint"))
semantic_rows <- bind_rows(lapply(fits, `[[`, "semantic"))

write_csv(single_rows, file.path(results_dir, "single_feature_coefficients.csv"))
write_csv(joint_rows, file.path(results_dir, "joint_coefficients.csv"))
if (nrow(semantic_rows) > 0) {
  write_csv(semantic_rows, file.path(results_dir, "joint_with_semantic_coefficients.csv"))
}
cat(sprintf("Wrote GLMM outputs to %s\n", results_dir))

