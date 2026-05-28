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

data_dir <- arg_value("--data-dir", "outputs/paper/semantic_datasets")
results_dir <- arg_value("--results-dir", "outputs/paper/r/semantics")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) stop(sprintf("No CSV files found in %s", data_dir))

fit_one <- function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  df <- df %>%
    mutate(
      source_prompt_id = factor(source_prompt_id),
      suffix_id = factor(suffix_id),
      target_prompt_id = factor(target_prompt_id),
      jailbroken = as.integer(jailbroken),
      cosine_sim_std = as.numeric(scale(cosine_similarity))
    )

  dimensionality <- unique(df$dimensionality)[[1]]
  if (dimensionality == "1d") {
    formula <- jailbroken ~ cosine_sim_std + (1 | target_prompt_id) + (1 | suffix_id)
  } else {
    formula <- jailbroken ~ cosine_sim_std + (1 | target_prompt_id) + (1 | source_prompt_id / suffix_id)
  }

  model <- glmer(
    formula,
    data = df,
    family = binomial,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
  r2_vals <- r2(model)
  fixed <- tidy(model, effects = "fixed") %>%
    mutate(
      file = basename(path),
      model_alias = unique(df$model_alias)[[1]],
      embedding = unique(df$embedding)[[1]],
      dimensionality = dimensionality,
      n = nrow(df),
      r2_marginal = r2_vals$R2_marginal,
      r2_conditional = r2_vals$R2_conditional
    )
  fixed
}

rows <- bind_rows(lapply(files, fit_one))
out_path <- file.path(results_dir, "semantic_coefficients.csv")
write_csv(rows, out_path)
cat(sprintf("Wrote %s\n", out_path))

