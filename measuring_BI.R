library(tuneR)
library(soundecology)
library(dplyr)
library(purrr)
library(tibble)

# ---- folder with WAVs ----
#wav_dir <- "D:/BARLT Localization Project/localization_06082025/localizationtrim"  # <-- change me
#wav_dir <-"D:/BARLT Localization Project/localization_05312025/localizationtrim_new"
wav_dir <-"D:/BARLT Localization Project/localization_06022025/localizationtrim"

# ---- match your BI settings ----
bi_min_freq <- 2000
bi_max_freq <- 8000
bi_fft_w <- 512

calc_bi_file <- function(path) {
  w <- tuneR::readWave(path)
  if (w@stereo) w <- tuneR::mono(w, which = "left")

  bi <- soundecology::bioacoustic_index(
    w,
    min_freq = bi_min_freq,
    max_freq = bi_max_freq,
    fft_w = bi_fft_w
  )
  as.numeric(bi$left_area)
}

wav_paths <- list.files(wav_dir, pattern = "\\.wav$", full.names = TRUE, recursive = TRUE)

bi_by_file <- tibble(path = wav_paths) %>%
  mutate(
    file = basename(path),
    BI = purrr::map_dbl(path, ~ tryCatch(calc_bi_file(.x), error = function(e) NA_real_))
  ) %>%
  filter(!is.na(BI))

# ---- summary stats across ALL recordings ----
avg_BI <- mean(bi_by_file$BI)
med_BI <- median(bi_by_file$BI)
min_BI <- min(bi_by_file$BI)
max_BI <- max(bi_by_file$BI)

tol <- 0.01 * abs(med_BI)

level <- case_when(
  abs(avg_BI - med_BI) <= tol ~ "medium",
  avg_BI < med_BI             ~ "low",
  TRUE                        ~ "high"
)

cat(sprintf(
  "Files used: %d\nBI min/median/max: %.3f / %.3f / %.3f\nAverage BI: %.3f (%s)\n",
  nrow(bi_by_file),
  min_BI, med_BI, max_BI,
  avg_BI, level
))

# save per-file BI for QA
readr::write_csv(bi_by_file, file.path(wav_dir, "BI_by_file.csv"))

# date : 05312025 --> high
    # which means threshold for VEER is 0.2
    # which means threshold for COYE is 0.4
    # which means threshold for MAWA is 0.75

# date : 06022025 --> 
    # which means threshold for VEER is 0.2
    # which means threshold for COYE is 0.4
    # which means threshold for MAWA is 0.75

# date : 06082025 --> 
   # which means threshold for VEER is 0.25
    # which means threshold for COYE is 0.4
    # which means threshold for MAWA is 0.7


