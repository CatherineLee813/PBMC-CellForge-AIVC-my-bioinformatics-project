setwd("D:/临时文件/AIVC/stimulated vs control/immune_alignment_expression_matrices/")
library(ggplot2)
library(readr)
library(dplyr)
library(data.table)
library(patchwork)
library(uwot)
library(Rtsne)
library(pheatmap)
library(RColorBrewer)

out_dir <- "scgen_visualization"

###training curve
df <- read_csv("scgen_out/training_curve.csv")

ggplot(df, aes(x=epoch)) +
  geom_line(aes(y=train_loss, color="Train"), size=1.1) +
  geom_line(aes(y=val_loss, color="Validation"), size=1.1) +
  scale_color_manual(values=c("Train"="#0072B2", "Validation"="#D55E00")) +
  theme_minimal(base_size=16) +
  labs(
    title="scGen Training Curve",
    y="Loss",
    color=""
  )


# Load latent space

latent <- fread(file.path("scgen_out/latents_umap_input.csv"))
latent$V33[latent$V33 == "stim"] <- "STIM"
latent_mat <- as.matrix(latent[, 1:32])
type <- latent$V33

latent_mat <- as.data.frame(latent_mat)

latent_mat[] <- lapply(latent_mat, function(x) {
  as.numeric(as.character(x))
})

latent_mat <- as.matrix(latent_mat)


#检查
#print(dim(latent_mat))
#print(sum(is.na(latent_mat)))
#sapply(latent_mat, class)


# Run UMAP

set.seed(42)
umap_res <- uwot::umap(latent_mat, n_neighbors = 30, min_dist = 0.3)
df_umap <- data.frame(
  UMAP1 = umap_res[,1],
  UMAP2 = umap_res[,2],
  stim  = type
)

#head(df_umap)

p1 <- ggplot(df_umap, aes(UMAP1, UMAP2, color = stim)) +
  geom_point(size = 1, alpha = 0.7) +
  theme_classic() +
  ggtitle("UMAP of latent_mu (32-dimensional)")

ggsave("scgen_out/R_latent_umap_fixed.png", p1, width = 6, height = 5, dpi = 150)

# Run t-SNE
set.seed(42)
tsne_res <- Rtsne(latent_mat, perplexity = 30)

df_tsne <- data.frame(
  tSNE1 = tsne_res$Y[,1],
  tSNE2 = tsne_res$Y[,2],
  stim  = type
)

p2 <- ggplot(df_tsne, aes(tSNE1, tSNE2, color = stim)) +
  geom_point(size = 1, alpha = 0.7) +
  theme_classic() +
  ggtitle("t-SNE of latent_mu")

ggsave("scgen_out/R_latent_tsne_fixed.png", p2, width = 6, height = 5, dpi = 150)

# Marker gene comparison (real vs predicted)
real_marker <- fread(file.path("scgen_out/marker_expression_real.csv"))
pred_marker <- fread(file.path("scgen_out/marker_expression_pred.csv"))

# merge on orig_idx
merged_marker <- merge(real_marker, pred_marker, by = "orig_idx", suffixes = c("_real", "_pred"))

# Extract marker list automatically
marker_genes <- gsub("_real", "", grep("_real$", colnames(merged_marker), value = TRUE))

for (g in marker_genes) {
  g_real <- paste0(g, "_real")
  g_pred <- paste0(g, "_pred")
  
  p <- ggplot(merged_marker, aes_string(x = g_real, y = g_pred)) +
    geom_point(alpha = 0.6, size = 1) +
    theme_bw() +
    ggtitle(paste("Marker expression:", g, "(real CTRL vs predicted STIM)")) +
    xlab("Real CTRL (log-normalized)") +
    ylab("Predicted STIM (log-normalized)")
  
  ggsave(file.path("scgen_out/R_marker_scatter.png"), p,
         width = 6, height = 4, dpi = 150)
}

# Heatmap of HVGs (stim vs control)
library(ComplexHeatmap)
library(circlize)
# 读取数据
heat_df <- fread("scgen_out/heatmap_input.csv")

expr_mat <- as.matrix(heat_df[, -ncol(heat_df), with = FALSE])
stim <- heat_df$stim

# log1p transform
expr_log <- log1p(expr_mat)

# Row Z-score
expr_scaled <- t(scale(t(expr_log)))
expr_scaled[is.na(expr_scaled)] <- 0

# Limit range
expr_trim <- expr_scaled
expr_trim[expr_trim > 2] <- 2
expr_trim[expr_trim < -2] <- -2

# Select top 40 most variable genes
gene_var <- apply(expr_log, 2, var)
top_genes <- names(sort(gene_var, decreasing = TRUE))[1:40]
expr_top <- expr_trim[, top_genes]

# Row names
rownames(expr_top) <- paste0("Cell_", seq_len(nrow(expr_top)))

# --------------------------------------
# Row annotation (stim is per cell)
# --------------------------------------
ha_row <- rowAnnotation(
  stim = stim,
  col = list(stim = c("CTRL"="#4daf4a", "STIM"="#e41a1c")),
  annotation_legend_param = list(title = "stim")
)

# Color function
col_fun <- colorRamp2(c(-2, 0, 2), c("#4575b4", "white", "#d73027"))

# --------------------------------------
# Heatmap (correct version)
# --------------------------------------
png("scgen_out/ComplexHeatmap_final_correct.png", width = 2400, height = 3200, res = 300)

ha_row +
  Heatmap(
    expr_top,
    name = "Z-score",
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    show_row_names = FALSE,
    show_column_names = TRUE,
    column_names_gp = gpar(fontsize = 10),
    heatmap_legend_param = list(
      at = c(-2, -1, 0, 1, 2),
      color_bar = "continuous",
      title = "Z-score"
    )
  )

dev.off()
