#!/usr/bin/env Rscript
# ----------------------------------------------------------------------------------
# Schaefer Atlas Visualization with ggseg (200 Parcels, 17-Network Version)
# Command-line version
# ----------------------------------------------------------------------------------

# ---- Parse command-line arguments -------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop(
    "\nUsage:\n  Rscript schaefer_plot.R <label_file> <data_dir> <output_dir> [colormap]\n\n",
    call. = FALSE
  )
}

labels_path    <- normalizePath(args[1],  mustWork = TRUE)
data_dir       <- normalizePath(args[2],  mustWork = TRUE)
output_dir     <- normalizePath(args[3],  mustWork = FALSE)
colormap_name  <- if (length(args) >= 4) args[4] else NULL
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Libraries -------------------------------------------------------------------
suppressPackageStartupMessages({
  library(ggplot2)
  library(ggseg)
  library(ggsegSchaefer)
  library(dplyr)
  library(viridis)
  library(RColorBrewer)
})

get_palette <- function(name, n = 256) {
  brewer_ok <- rownames(RColorBrewer::brewer.pal.info)
  viridis_ok <- c("magma","inferno","plasma","viridis","cividis","mako","rocket","turbo")
  if (is.null(name)) return(viridis(n))  # default

  if (name %in% brewer_ok) {
    return(colorRampPalette(RColorBrewer::brewer.pal(9, name))(n))
  } else if (tolower(name) %in% viridis_ok) {
    return(viridis(n, option = tolower(name)))
  } else {
    stop(sprintf("Unknown colormap '%s'. Use a Brewer palette (e.g., 'RdBu') or a viridis option (e.g., 'magma').", name))
  }
}

# ---- Plotting helper --------------------------------------------------------------
generate_ggseg_plot <- function(atlas, fill_variable,
                                limits = NULL, breaks = NULL,
                                output_path, filename,
                                midpoint_mode = FALSE, decimals = FALSE,
                                colormap = c("white", "red")) {
  format_fn <- if (decimals)
                 scales::number_format(accuracy = 0.01)
               else
                 scales::number_format(accuracy = 1)
  
  fill_values <- atlas$data[[fill_variable]]
  fill_values <- fill_values[!is.na(fill_values)]
  
  if (is.null(limits)) limits <- range(fill_values, na.rm = TRUE)
  if (is.null(breaks)) breaks <- pretty(limits, n = 5)
  
  fill_scale <- if (midpoint_mode) {
    scale_fill_gradient2(
      low = colormap[1], mid = "white", high = tail(colormap, 1),
      midpoint = 0, limits = limits, breaks = breaks,
      na.value = "grey90", labels = format_fn)
  } else if (length(colormap) == 2) {
    scale_fill_gradient(
      low = colormap[1], high = colormap[2],
      limits = limits, breaks = breaks,
      na.value = "grey90", labels = format_fn)
  } else {
    scale_fill_gradientn(
      colours = colormap,
      limits = limits, breaks = breaks,
      na.value = "grey90", labels = format_fn)
  }
  
  p <- ggseg(
      atlas = atlas, colour = "black", position = "stacked",
      mapping = aes_string(fill = fill_variable)
    ) +
    fill_scale +
    theme_void() +
    theme(
      plot.background   = element_rect(fill = "transparent", colour = NA),
      panel.background  = element_rect(fill = "transparent", colour = NA),
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.key        = element_rect(fill = "transparent", colour = NA),
      legend.title      = element_blank()
    )
  
  ggsave(
    filename = file.path(output_path, paste0(filename, ".svg")),
    plot     = p,
    width    = 6, height = 6, units = "in", dpi = 300, bg = "transparent",
    device = "svg"
  )
}

# ---- Load atlas & labels ----------------------------------------------------------
atlas_object    <- schaefer17_200
schaefer_labels <- read.csv(labels_path, header = TRUE, sep = "")

# ---- Constants you may still want to expose --------------------------------------
#plot_limits <- c(0, 800)   # adjust as needed

# ---- Process each CSV -------------------------------------------------------------
csv_files <- list.files(
  data_dir, pattern = "\\.csv$", full.names = TRUE
)

# Determine global color limits across all CSV files
all_vals <- c()
for (fp in csv_files) {
  vals <- read.csv(fp, header = TRUE)$X0
  all_vals <- c(all_vals, vals[!is.na(vals)])
}
global_limits <- range(all_vals, na.rm = TRUE)
midpoint_mode <- global_limits[1] < 0 && global_limits[2] > 0
if (!is.null(colormap_name) && !midpoint_mode) {
  colormap <- get_palette(colormap_name)
} else if (midpoint_mode) {
  colormap <- c("red", "white", "blue")
} else {
  colormap <- viridis(256)
}

for (file_path in csv_files) {
  data_filename <- tools::file_path_sans_ext(basename(file_path))

  schaefer_data <- read.csv(file_path, header = TRUE)$X0

  values <- numeric(202)
  values[1:2]   <- NA_real_      # medial walls
  values[3:202] <- schaefer_data

  brain_data <- data.frame(region = schaefer_labels, brain_data = values)

  atlas_polys <- atlas_object$data[4]
  colnames(atlas_polys) <- c("region", "geometry")

  df_merged <- atlas_polys %>%
               left_join(brain_data, by = "region")

  atlas_object$data <- atlas_object$data %>%
                       mutate(brain_data = df_merged$brain_data)

  # Overwrite the colormap to match local limits
  global_limits <- range(df_merged$brain_data, na.rm = TRUE)
  midpoint_mode <- global_limits[1] < 0 && global_limits[2] > 0
  if (!is.null(colormap_name) && !midpoint_mode) {
    colormap <- get_palette(colormap_name)
  } else if (midpoint_mode) {
    colormap <- c("red", "white", "blue")
  } else {
    colormap <- viridis(256)
  }

  # Emit colorbar limits for this plot to stdout in a parseable format
  cmin <- formatC(global_limits[1], digits = 10, format = "g")
  cmax <- formatC(global_limits[2], digits = 10, format = "g")
  cat(sprintf("CBAR file=%s min=%s max=%s\n", data_filename, cmin, cmax))

  generate_ggseg_plot(
    atlas         = atlas_object,
    fill_variable = "brain_data",
    limits        = global_limits,
    output_path   = output_dir,
    filename      = data_filename,
    midpoint_mode = midpoint_mode,
    decimals      = TRUE,
    colormap      = colormap
  )
}
