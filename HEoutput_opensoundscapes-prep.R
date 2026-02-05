## ---------------------------------------------------------------
## Converting HawkEars_labels.csv -> aru_coords.csv + detections.csv
## Adapted from Erica's script for Megan's BBMP localization grid
## ---------------------------------------------------------------

library(tidyverse)
library(stringr)

## ---- paths ----------------------------------------------------
# HawkEars output (your CSV)
labels_path <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_2thresh_VEER/HawkEars_labels.csv"

# Folder where the trimmed wav files live
wav_dir <- "D:/BARLT Localization Project/localization_05312025/localizationtrim_new"

# RTK / site coordinates (your grid)
sites_path <- "D:/BARLT Localization Project/LocalizationSites_CWS_2025.csv"

# Output files for opensoundscapes
aru_coords_out   <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_2thresh_VEER/aru_coords.csv"
detections_out   <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_2thresh_VEER/detections_all_species.csv"

# OPTIONAL: detections for one species
detections_veer_out <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_2thresh_VEER/detections_VEER.csv"

## ---- read HawkEars labels ------------------------------------
all_data <- read.csv(labels_path, stringsAsFactors = FALSE)

# If you want to apply a confidence threshold, set it here:
min_score <- 0.7
all_data <- all_data |> 
  filter(score >= min_score)

# Add full filepath, ARU ID, and harmonize column names
all_data <- all_data |>
  mutate(
    file      = file.path(wav_dir, filename),
    aru_id    = sub("_.*", "", filename),   # e.g., "L1N1E1" from "L1N1E1_S2025..."
    tag_start = as.numeric(start_time),
    tag_end   = as.numeric(end_time),
    species   = class_code                  # use short code (VEER, COYE, etc.)
  )

## quick sanity checks
unique(all_data$aru_id)[1:10]
head(all_data$file)
head(all_data$species)

## ---- build aru_coords.csv ------------------------------------
sites <- read.csv(sites_path, stringsAsFactors = FALSE)

# rename to match Erica's expected x/y
sites_clean <- sites |>
  transmute(
    aru_id = SiteID,
    x = Longitude,
    y = Latitude
  )

# one row per recording file, joined to ARU coordinates
aru_coords <- all_data |>
  distinct(file, aru_id) |>
  left_join(sites_clean, by = "aru_id")

# check for any ARUs that didn't match
aru_coords |> filter(is.na(x) | is.na(y)) |> distinct(aru_id)

# keep only the columns opensoundscapes needs
aru_coords_out_df <- aru_coords |>
  select(file, x, y) |>
  distinct()

write.csv(aru_coords_out_df, aru_coords_out, row.names = FALSE)

## ---- build detections.csv (3 s bins, all species) ------------

# find max end time across all tags to define binning
global_max_end <- max(all_data$tag_end, na.rm = TRUE)

# 3-second bins
all_bins <- tibble(
  bin_start = seq(0, global_max_end, by = 3)
) |>
  mutate(bin_end = bin_start + 3)

# ARU × species combos actually present in the HawkEars output
aru_sp <- all_data |>
  distinct(aru_id, species)

# expand to ARU × species × time bins
bins_sp <- aru_sp |>
  crossing(all_bins)

# mark bins where that species is present on that ARU
presence_df <- bins_sp |>
  left_join(all_data, by = c("aru_id", "species")) |>
  # keep only bins overlapping with at least one tag
  filter(!(bin_end <= tag_start | bin_start >= tag_end)) |>
  distinct(file, species, bin_start, bin_end) |>
  mutate(present = 1)

# tidy + pivot to wide
final_wide <- presence_df |>
  select(file, bin_start, bin_end, species, present) |>
  distinct() |>
  pivot_wider(
    names_from  = species,   # columns: VEER, COYE, NAWA, ...
    values_from = present,
    values_fill = 0
  ) |>
  arrange(file, bin_start) |>
  rename(
    start_time = bin_start,
    end_time   = bin_end
  )

write.csv(final_wide, detections_out, row.names = FALSE)

## ---- OPTIONAL: detections for a single species (e.g., VEER) ---
if ("VEER" %in% names(final_wide)) {
  detections_veer <- final_wide |>
    select(file, start_time, end_time, VEER)
  write.csv(detections_veer, detections_veer_out, row.names = FALSE)
}
