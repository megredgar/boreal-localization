## ---------------------------------------------------------------
## Localization prep: CONW (Connecticut Warbler)
## Converts HawkEars detections -> aru_coords.csv + detections_CONW.csv
## for OpenSoundscape localization
## ---------------------------------------------------------------
library(tidyverse)
library(stringr)

## ---- paths ----------------------------------------------------
labels_path <- "D:/BARLT Localization Project/localization_05312025/hawkears_lowthresh/HawkEars_labels.csv"
wav_dir     <- "D:/BARLT Localization Project/localization_05312025/localizationtrim_new"
sites_path  <- "D:/BARLT Localization Project/LocalizationSites_CWS_2025.csv"

out_dir <- "D:/BARLT Localization Project/localization_05312025/hawkears_0_7_CONW"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

species_code <- "CONW"
threshold    <- 0.7

## ---- read + filter for this species only ----------------------
raw <- read.csv(labels_path, stringsAsFactors = FALSE)

det <- raw |>
  mutate(
    species   = class_code,
    aru_id    = str_extract(filename, "L\\d+N\\d+E\\d+"),
    file      = file.path(wav_dir, filename),
    tag_start = as.numeric(start_time),
    tag_end   = as.numeric(end_time)
  ) |>
  filter(!is.na(aru_id), aru_id != "") |>
  filter(species == species_code) |>
  filter(score >= threshold)

cat(sprintf("CONW detections after threshold (%.2f): %d\n", threshold, nrow(det)))

## ---- build aru_coords.csv ------------------------------------
sites_clean <- read.csv(sites_path, stringsAsFactors = FALSE) |>
  transmute(
    aru_id = SiteID,
    x = Longitude,
    y = Latitude
  )

aru_coords <- det |>
  distinct(file, aru_id) |>
  left_join(sites_clean, by = "aru_id") |>
  select(file, x, y) |>
  distinct()

## check for missing coordinates
missing <- aru_coords |> filter(is.na(x) | is.na(y))
if (nrow(missing) > 0) {
  warning(sprintf("%d files have missing coordinates — check SiteID matching", nrow(missing)))
  print(missing)
}

aru_coords_path <- file.path(out_dir, "aru_coords.csv")
write.csv(aru_coords, aru_coords_path, row.names = FALSE)

## ---- build detections_CONW.csv (3 s bins) ---------------------
max_end <- max(det$tag_end, na.rm = TRUE)

bins <- tibble(bin_start = seq(0, max_end, by = 3)) |>
  mutate(bin_end = bin_start + 3)

## get all unique files for this species
all_files <- det |> distinct(file)

## create full grid: every file x every bin
file_bins <- all_files |> crossing(bins)

## flag bins with a detection
hits <- det |>
  crossing(bins) |>
  filter(!(bin_end <= tag_start | bin_start >= tag_end)) |>
  distinct(file, bin_start, bin_end) |>
  mutate(!!species_code := 1L)

## join back to full grid so absent bins = 0
detections <- file_bins |>
  left_join(hits, by = c("file", "bin_start", "bin_end")) |>
  replace_na(setNames(list(0L), species_code)) |>
  arrange(file, bin_start) |>
  rename(start_time = bin_start, end_time = bin_end)

detections_path <- file.path(out_dir, paste0("detections_", species_code, ".csv"))
write.csv(detections, detections_path, row.names = FALSE)

## ---- summary --------------------------------------------------
cat("\n--- CONW summary ---\n")
cat(sprintf("  ARU coords file: %s  [exists: %s]\n", aru_coords_path, file.exists(aru_coords_path)))
cat(sprintf("  Detections file: %s  [exists: %s]\n", detections_path, file.exists(detections_path)))
cat(sprintf("  Unique files:    %d\n", n_distinct(det$file)))
cat(sprintf("  Unique ARUs:     %d\n", n_distinct(det$aru_id)))
cat(sprintf("  Detection bins:  %d / %d total bins\n",
            sum(detections[[species_code]] == 1), nrow(detections)))