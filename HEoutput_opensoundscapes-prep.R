## ---------------------------------------------------------------
## Converting HawkEars_labels.csv -> aru_coords.csv + detections.csv
## Adapted from Erica's script for Megan's BBMP localization grid
## ---------------------------------------------------------------
library(tidyverse)
library(stringr)

## ---- paths ----------------------------------------------------
labels_path <- "D:/BARLT Localization Project/localization_05312025/hawkears_lowthresh/HawkEars_labels.csv"
wav_dir     <-"D:/BARLT Localization Project/localization_05312025/localizationtrim_new"  # <- adjust if different
sites_path  <- "D:/BARLT Localization Project/LocalizationSites_CWS_2025.csv"

out_conw <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW"
out_ambi <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_7_AMBI"
out_alfl<- "D:/BARLT Localization Project/localization_05312025/hawkears_0_7_ALFL"

aru_coords_out <- file.path(out_conw, "aru_coords.csv")
detections_out <- file.path(out_conw, "detections_all_species.csv")

detections_conw_out<- file.path(out_conw, "detections_CONW.csv")
detections_alfl_out<- file.path(out_alfl, "detections_ALFL.csv")
detections_ambi_out<- file.path(out_ambi, "detections_AMBI.csv")

## ensure folders exist
dir.create(out_veer, recursive = TRUE, showWarnings = FALSE)
dir.create(out_coye, recursive = TRUE, showWarnings = FALSE)
dir.create(out_mawa, recursive = TRUE, showWarnings = FALSE)

## ---- read + threshold ----------------------------------------
#thr <- c(VEER = 0.2, COYE = 0.4, MAWA = 0.75)
thr <- c(AMBI = 0.7, ALFL = 0.7, CONW = 0.7)
all_data <- read.csv(labels_path, stringsAsFactors = FALSE) |>
  mutate(
    species   = class_code,
    aru_id    = str_extract(filename, "L\\d+N\\d+E\\d+"),
    file      = file.path(wav_dir, filename),
    tag_start = as.numeric(start_time),
    tag_end   = as.numeric(end_time)
  ) |>
  filter(!is.na(aru_id), aru_id != "") |>
  filter(species %in% names(thr)) |>
  filter(score >= unname(thr[species]))

## ---- build aru_coords.csv ------------------------------------
sites_clean <- read.csv(sites_path, stringsAsFactors = FALSE) |>
  transmute(
    aru_id = SiteID,
    x = Longitude,
    y = Latitude
  )

aru_coords_out_df <- all_data |>
  distinct(file, aru_id) |>
  left_join(sites_clean, by = "aru_id") |>
  select(file, x, y) |>
  distinct()

write.csv(aru_coords_out_df, aru_coords_out, row.names = FALSE)

## ---- build detections_all_species.csv (3 s bins) -------------
global_max_end <- max(all_data$tag_end, na.rm = TRUE)

all_bins <- tibble(bin_start = seq(0, global_max_end, by = 3)) |>
  mutate(bin_end = bin_start + 3)

aru_sp <- all_data |>
  distinct(aru_id, species)

bins_sp <- aru_sp |>
  crossing(all_bins)

presence_df <- bins_sp |>
  left_join(all_data, by = c("aru_id", "species")) |>
  filter(!(bin_end <= tag_start | bin_start >= tag_end)) |>
  distinct(file, species, bin_start, bin_end) |>
  mutate(present = 1)

final_wide <- presence_df |>
  select(file, bin_start, bin_end, species, present) |>
  distinct() |>
  pivot_wider(
    names_from  = species,
    values_from = present,
    values_fill = 0
  ) |>
  arrange(file, bin_start) |>
  rename(start_time = bin_start, end_time = bin_end)

write.csv(final_wide, detections_out, row.names = FALSE)

## ---- write per-species detections (always write a file) -------
write_species <- function(sp, out_path) {
  dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
  if (sp %in% names(final_wide)) {
    out <- final_wide |> select(file, start_time, end_time, all_of(sp))
  } else {
    out <- final_wide |> select(file, start_time, end_time) |> mutate(!!sp := 0)
  }
  write.csv(out, out_path, row.names = FALSE)
}

write_species("AMBI", detections_ambi_out)
write_species("CONW", detections_conw_out)
write_species("ALFL", detections_alfl_out)

## ---- quick “did it save?” checks ------------------------------
file.exists(aru_coords_out)
file.exists(detections_out)
file.exists(detections_veer_out)
file.exists(detections_coye_out)
file.exists(detections_mawa_out)
