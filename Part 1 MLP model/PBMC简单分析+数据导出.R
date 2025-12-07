library(Seurat)
library(dplyr)


pbmc.data <- Read10X(data.dir = "D:/临时文件/AIVC/filtered_gene_bc_matrices/hg19")
#Initialize the Seurat object with the raw non-normalized data
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k", min.cells = 3, min.features = 200)

pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

pbmc <- NormalizeData(pbmc)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
pbmc <- ScaleData(pbmc, features = VariableFeatures(pbmc))
pbmc <- RunPCA(pbmc, features = VariableFeatures(pbmc))
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)
pbmc <- RunUMAP(pbmc, dims = 1:10)
DimPlot(pbmc, reduction = "umap", label = TRUE)




# 兼容性更强的导出脚本（Seurat v3/v4/v5 通用）
library(Seurat)

# 假设你的 Seurat 对象名为 pbmc （如不是请替换）
# pbmc <- ... 

# --- 诊断信息（可选，帮你检查对象结构） ---
cat("Assays in pbmc:\n")
print(Assays(pbmc))         # 显示有哪些 assay
cat("\nSlot names of RNA assay (if exists):\n")
if ("RNA" %in% Assays(pbmc)) {
  print(slotNames(pbmc[["RNA"]]))
} else {
  warning("对象中没有 'RNA' assay，请确认你的 assay 名称（例如 'RNA' 或 'SCT'）")
}

# --- 安全地获取细胞列表（用于保持一致顺序） ---
# 优先选取 PCA embeddings 的行名（通常为 cell names），否则用 pbmc 列名
cell_order <- NULL
if ("pca" %in% names(pbmc@reductions)) {
  cell_order <- rownames(Embeddings(pbmc, "pca"))
} else if ("umap" %in% names(pbmc@reductions)) {
  cell_order <- rownames(Embeddings(pbmc, "umap"))
} else {
  cell_order <- colnames(pbmc)  # fallback
}
cat("Using", length(cell_order), "cells (first 5):", head(cell_order,5), "\n")

# --- 1) 导出 normalized data (slot = "data") ---
# Use GetAssayData to be robust
assay_name <- "RNA"
if (! (assay_name %in% Assays(pbmc))) {
  # 如果没有 RNA，尝试取第一个 assay
  assay_name <- Assays(pbmc)[1]
  warning(paste0("没有检测到 'RNA' assay，改为使用第一个 assay: ", assay_name))
}

# 检查是否存在 slot "data"
has_data_slot <- TRUE
# 尝试获取（如果不存在会抛错，用 tryCatch 捕捉）
norm_counts_mat <- tryCatch({
  as.matrix(GetAssayData(pbmc, assay = assay_name, slot = "data"))
}, error = function(e) {
  message("无法直接用 slot='data' 读取（可能 slot 名称不同），尝试用 'counts' 或警告。")
  has_data_slot <<- FALSE
  NULL
})

# 如果 data 不存在，尝试用 counts 作为替代（或先 NormalizeData 再导出）
if (is.null(norm_counts_mat)) {
  # 如果没有 normalized data，先确保 NormalizedData 存在：可以用 NormalizeData(pbmc) 生成
  if (!("data" %in% slotNames(pbmc[[assay_name]]))) {
    message("尝试导出 counts；如果你需要 normalized data，请运行 NormalizeData(pbmc) 再重试。")
    norm_counts_mat <- tryCatch(as.matrix(GetAssayData(pbmc, assay = assay_name, slot = "counts")),
                                error = function(e) { stop("counts 也无法读取，请检查 assay/slots。") })
  }
}

# 现在导出（按 cell_order 重新排序列）
if (!is.null(norm_counts_mat)) {
  # norm_counts_mat 的行通常是基因，列是细胞
  # 保证列顺序和 cell_order 一致（取交集并按 cell_order 排序）
  common_cells <- intersect(colnames(norm_counts_mat), cell_order)
  if (length(common_cells) == 0) stop("找不到共同细胞名，请检查 pbmc 对象的 colnames 与 reductions 的 rownames 是否一致。")
  norm_counts_mat <- norm_counts_mat[, cell_order[cell_order %in% common_cells], drop = FALSE]
  write.csv(norm_counts_mat, file = "norm_counts.csv")
  cat("已导出 norm_counts.csv (genes x cells)。\n")
}

# --- 2) 导出 scale.data（如果有） ---
# 获取 scale.data 层
library(Matrix)

scaled_mat <- GetAssayData(pbmc, assay="RNA", slot="scale.data")

# 确保转换为稀疏矩阵
scaled_mat <- as(scaled_mat, "dgCMatrix")

write.csv(as.matrix(scaled_mat), "scaled_data.csv")

# --- 3) 导出 variable genes 列表 ---
if (!is.null(VariableFeatures(pbmc))) {
  var_genes <- VariableFeatures(pbmc)
  write.csv(var_genes, file = "variable_genes.csv", row.names = FALSE)
  cat("已导出 variable_genes.csv。\n")
} else {
  message("VariableFeatures(pbmc) 为空。请确认是否已运行 FindVariableFeatures()。")
}

# --- 4) 导出 PCA embeddings（如果存在） ---
if ("pca" %in% names(pbmc@reductions)) {
  pca_emb <- Embeddings(pbmc, "pca")
  # 按 cell_order 排序行
  pca_emb <- pca_emb[cell_order[cell_order %in% rownames(pca_emb)], , drop = FALSE]
  write.csv(pca_emb, file = "pca_embeddings.csv")
  cat("已导出 pca_embeddings.csv（cells x PCs）。\n")
} else {
  message("未检测到 PCA（请先运行 RunPCA(pbmc, features=VariableFeatures(pbmc))）。")
}

# --- 5) 导出 UMAP embeddings（如果存在） ---
if ("umap" %in% names(pbmc@reductions)) {
  umap_emb <- Embeddings(pbmc, "umap")
  umap_emb <- umap_emb[cell_order[cell_order %in% rownames(umap_emb)], , drop = FALSE]
  write.csv(umap_emb, file = "umap_embeddings.csv")
  cat("已导出 umap_embeddings.csv（cells x 2）。\n")
} else {
  message("未检测到 UMAP（请先运行 RunUMAP(pbmc, dims=...)）。")
}

# --- 6) 导出聚类 / 标签信息（兼容多种写法） ---
# 尽量使用 Idents() 来获取当前活跃 identity
idents <- NULL
try({
  idents <- Idents(pbmc)
}, silent = TRUE)

if (is.null(idents) || length(idents) == 0) {
  # fallback：尝试 pbmc$seurat_clusters
  if ("seurat_clusters" %in% colnames(pbmc@meta.data)) {
    idents <- pbmc$seurat_clusters
  } else {
    stop("无法获取细胞标签。请确认已经运行 FindClusters() 并且有相关 meta.data。")
  }
}

# 将 idents 转为 dataframe，并按 cell_order 排序
idents_df <- data.frame(cell = names(idents), cluster = as.character(as.vector(idents)), stringsAsFactors = FALSE)
idents_df <- idents_df[match(cell_order, idents_df$cell), , drop = FALSE]   # 保持和 embeddings 的顺序
# 移除 NA 行（如果有）
idents_df <- idents_df[!is.na(idents_df$cell), ]

write.csv(idents_df, file = "clusters.csv", row.names = FALSE)
cat("已导出 clusters.csv（cell, cluster）。\n")

# --- 结束 ---
cat("全部导出结束。请检查生成的 CSV 是否位于当前工作目录（getwd()）。\n")

