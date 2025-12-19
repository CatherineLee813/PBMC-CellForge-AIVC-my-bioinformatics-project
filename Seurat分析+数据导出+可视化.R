# seurat_integration_and_export.R
# 目的：基于 Kang et al. 数据（stim vs ctrl），使用 Seurat v3 完成整合、
# 可视化，并导出训练集/测试集（表达矩阵 + metadata）。
# 运行：Rscript seurat_integration_and_export.R
#
# 说明：脚本默认读取 data/immune_control_expression_matrix.txt.gz
#      和 data/immune_stimulated_expression_matrix.txt.gz（与教程一致）。
#      输出在 output/ 目录下。
setwd("D:/临时文件/AIVC/stimulated vs control/immune_alignment_expression_matrices")

library(Seurat)
library(Matrix)
library(cowplot)  # 用于拼图
library(dplyr)
library(ggplot2)

# --------------------------
# 参数 / 路径
# --------------------------

outdir <- "output"
ctrl_file <- file.path( "immune_control_expression_matrix.txt.gz")
stim_file <- file.path( "immune_stimulated_expression_matrix.txt.gz")

set.seed(42) # 保持可复现

# --------------------------
# 1) 读入并构建 Seurat objects
# --------------------------
message("Loading data...")
ctrl.data <- read.table(file = ctrl_file, sep = "\t", header = TRUE, row.names = 1)
stim.data <- read.table(file = stim_file, sep = "\t", header = TRUE, row.names = 1)

# Create Seurat objects
ctrl <- CreateSeuratObject(counts = ctrl.data, project = "IMMUNE_CTRL", min.cells = 5)
ctrl$stim <- "CTRL"
ctrl <- subset(ctrl, subset = nFeature_RNA > 500)
ctrl <- NormalizeData(ctrl, verbose = FALSE)
ctrl <- FindVariableFeatures(ctrl, selection.method = "vst", nfeatures = 2000)

stim <- CreateSeuratObject(counts = stim.data, project = "IMMUNE_STIM", min.cells = 5)
stim$stim <- "STIM"
stim <- subset(stim, subset = nFeature_RNA > 500)
stim <- NormalizeData(stim, verbose = FALSE)
stim <- FindVariableFeatures(stim, selection.method = "vst", nfeatures = 2000)

# --------------------------
# 2) 找 anchors 并整合数据 (Integration)
# --------------------------
message("Finding integration anchors...")
immune.anchors <- FindIntegrationAnchors(object.list = list(ctrl, stim), dims = 1:20)
immune.combined <- IntegrateData(anchorset = immune.anchors, dims = 1:20)

# --------------------------
# 3) 统一分析流程：Scale, PCA, UMAP, Clustering
# --------------------------
DefaultAssay(immune.combined) <- "integrated"
immune.combined <- ScaleData(immune.combined, verbose = FALSE)
immune.combined <- RunPCA(immune.combined, npcs = 30, verbose = FALSE)
immune.combined <- RunUMAP(immune.combined, reduction = "pca", dims = 1:20)
immune.combined <- FindNeighbors(immune.combined, reduction = "pca", dims = 1:20)
immune.combined <- FindClusters(immune.combined, resolution = 0.5)

# 保存 Seurat object（RDS）
saveRDS(immune.combined, file = file.path( "immune_combined_seurat_v3.rds"))

# --------------------------
# 4) 可视化并保存图片
# --------------------------
p1 <- DimPlot(immune.combined, reduction = "umap", group.by = "stim") + ggtitle("Condition (stim vs ctrl)")
p2 <- DimPlot(immune.combined, reduction = "umap", label = TRUE, repel = TRUE) + ggtitle("Clusters")
p_split <- DimPlot(immune.combined, reduction = "umap", split.by = "stim", label = FALSE)
png(filename = file.path(out_dir, "umap_stim_vs_ctrl.png"), width = 1400, height = 600)
plot_grid(p1, p2)
dev.off()
ggsave(filename = file.path(out_dir, "umap_split_by_stim.png"), plot = p_split, width = 12, height = 6)

# FeaturePlot 示例（保存一张）
FeaturePlot(immune.combined, features = c("CD3D","GNLY","MS4A1","CD14"), min.cutoff = "q9")
ggsave(file.path(out_dir, "featureplot_example.png"), width = 12, height = 8)

# --------------------------
# 5) 识别 conserved markers（按教程）
# --------------------------
# 使用 RNA assay 来找在分群中对两个条件都保守的 marker
# Ensure RNA assay is used for DE
DefaultAssay(immune.combined) <- "RNA"

# Join layers for DE tests
immune.combined <- JoinLayers(immune.combined)

cluster_to_check <- 7

if (as.character(cluster_to_check) %in% levels(Idents(immune.combined))) {
  nk.markers <- FindConservedMarkers(
    immune.combined,
    ident.1 = cluster_to_check,
    grouping.var = "stim",
    verbose = FALSE
  )
  
  write.csv(
    nk.markers,
    file = file.path(out_dir, paste0("conserved_markers_cluster", cluster_to_check, ".csv"))
  )
}


# --------------------------
# 6) 对每个细胞类型比较条件差异（示例：B cells）
# --------------------------
# 假设你已经为 cluster 做了注释，这里我们先把 Idents 设为当前 cluster
# （如果你手动注释了细胞类型，可将 Idents 设置为 celltype）
Idents(immune.combined) <- Idents(immune.combined) # 保持当前 cluster idents
# 举例：对比 B_STIM vs B_CTRL（确保 B cluster 存在）
# 下面仅示例，如果没有 'B' 这个 label，跳过
if ("B" %in% levels(Idents(immune.combined))) {
  immune.combined$celltype.stim <- paste(Idents(immune.combined), immune.combined$stim, sep = "_")
  immune.combined$celltype <- Idents(immune.combined)
  Idents(immune.combined) <- "celltype.stim"
  b.interferon.response <- FindMarkers(immune.combined, ident.1 = "B_STIM", ident.2 = "B_CTRL", verbose = FALSE)
  write.csv(b.interferon.response, file = file.path(out_dir, "B_cell_interferon_vs_ctrl_markers.csv"))
}

# --------------------------
# 7) 为 ML 导出训练集/测试集
#    - 我们提供分层抽样（按 cluster 保证比例）80/20 split
#    - 导出 raw counts (sparse Matrix Market), normalized data (log-normalized), 以及 metadata (CSV)
# --------------------------
message("Preparing train/test split and exporting data...")

# 使用 RNA 计数矩阵（raw counts）与 NormalizeData 后的 data（log1p）作为选择
DefaultAssay(immune.combined) <- "RNA"

# 提取表达矩阵（raw counts）和归一化数据（data slot）：
# raw counts:
raw_counts <- GetAssayData(immune.combined, assay = "RNA", slot = "counts")
# normalized (log-normalized)：
# 如果在最初我们对单个对象做了 NormalizeData，那么 RNA@data 里应该有 log-normalized values
norm_data <- GetAssayData(immune.combined, assay = "RNA", slot = "data")

# meta 数据（含 cluster, stim, cell barcodes, nFeature, nCount 等）
meta <- immune.combined@meta.data
meta$barcode <- rownames(meta)

# 分层抽样函数（per-cluster 80/20）
train_frac <- 0.8
set.seed(42)
meta <- meta %>% mutate(cluster = as.character(Idents(immune.combined)))
train_barcodes <- meta %>%
  group_by(cluster) %>%
  sample_frac(train_frac) %>%
  pull(barcode)
test_barcodes <- setdiff(meta$barcode, train_barcodes)

length_train <- length(train_barcodes)
length_test <- length(test_barcodes)
message(sprintf("Train cells: %d ; Test cells: %d", length_train, length_test))

# 保存 metadata 列表
write.csv(meta, file = file.path(out_dir, "metadata_all_cells.csv"), row.names = FALSE)
write.csv(meta[meta$barcode %in% train_barcodes, ], file = file.path(out_dir, "metadata_train.csv"), row.names = FALSE)
write.csv(meta[meta$barcode %in% test_barcodes, ], file = file.path(out_dir, "metadata_test.csv"), row.names = FALSE)

# 导出表达矩阵（稀疏 MatrixMarket 格式）-- raw counts
library(Matrix)
writeMM(raw_counts[, train_barcodes], file = file.path(out_dir, "raw_counts_train.mtx"))
writeMM(raw_counts[, test_barcodes], file = file.path(out_dir, "raw_counts_test.mtx"))
# 导出基因/细胞名称
write.csv(rownames(raw_counts), file = file.path(out_dir, "genes_raw_counts.csv"), row.names = FALSE)
write.csv(colnames(raw_counts[, train_barcodes]), file = file.path(out_dir, "barcodes_train_raw_counts.csv"), row.names = FALSE)
write.csv(colnames(raw_counts[, test_barcodes]), file = file.path(out_dir, "barcodes_test_raw_counts.csv"), row.names = FALSE)

# 导出归一化数据（稠密/稀疏视情况）
writeMM(as(raw_counts[, train_barcodes], "dgCMatrix"), file = file.path(out_dir, "raw_counts_train_sparse.mtx"))
# 如果想要导出 normalized (log) matrix，可以选择导出为 CSV（注意会大）
lognorm_train <- as.matrix(norm_data[, train_barcodes])
lognorm_test  <- as.matrix(norm_data[, test_barcodes])
write.csv(lognorm_train, file = file.path(out_dir, "lognorm_train.csv"))
write.csv(lognorm_test, file = file.path(out_dir, "lognorm_test.csv"))

# 另外保存 Seurat 对象的 train/test 子集（rds），便于后续检索
immune.train <- subset(immune.combined, cells = train_barcodes)
immune.test  <- subset(immune.combined, cells = test_barcodes)
saveRDS(immune.train, file = file.path(out_dir, "immune_train_seurat.rds"))
saveRDS(immune.test,  file = file.path(out_dir, "immune_test_seurat.rds"))

message("All exports finished. Output directory:")
print(normalizePath(out_dir))

# --------------------------
# 8) 建议 & 注意事项（脚本内注释）
# --------------------------
# - 训练/测试数据导出时要注意你后续模型的输入格式：稀疏矩阵 (MTX + genes + barcodes) 是很多 scRNA 工具/框架支持的。
# - 如果你的 AI 虚拟细胞模型需要批次信息或条件标签（stim），可以在 metadata_train/test.csv 中找到 stim 列。
# - 若你想保存更少的特征（例如只保留 top variable genes），可以在导出前筛选 gene 列： e.g. VariableFeatures(immune.combined) 返回的基因集。
# - 若你要训练分类器去区分 cell types，请先基于 conserved markers 或 manual annotation 给 clusters 标注 cell type（RenameIdents）。
# - 若想导出 scaled data（用于某些 ML 算法）可以用 GetAssayData(slot="scale.data")（在 ScaleData 之后）。
#
# 结束
sessionInfo()
