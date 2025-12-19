#finished by Catherine Li 2025/11/19
library(dplyr)
library(Seurat)
library(patchwork)

#设置工作环境
setwd("D:/临时文件/AIVC")
#load the dataset
pbmc.data <- Read10X(data.dir = "D:/临时文件/AIVC/filtered_gene_bc_matrices/hg19")
#Initialize the Seurat object with the raw non-normalized data
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k", min.cells = 3, min.features = 200)
#pbmc

#质控(筛选掉表现出严重线粒体污染的低质量/濒死细胞)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
#Visualize QC metrics as a violin plot
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)
#由图可知，需要过滤掉nFeature>2500或<200者以及线粒体数量大于5%的细胞
plot1 <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
#plot1+plot2
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)


#---Normalization---
#默认情况下，采用全局缩放归一化方法“LogNormalize”
#LogNormalize : 首先将每个细胞的特征表达测量值除以总表达量，
#然后乘以一个缩放因子（默认为 10,000），最后对结果进行对数变换
#在 Seurat v5 中，归一化后的值存储在pbmc[["RNA"]]$data

pbmc <- NormalizeData(pbmc)  #此时method默认为"Lognormalize"，scale.factor默认为10000
#现在有时候也用SCTransform()，容易PCA投射不准

#---特征选择---
#FindVariableFeatures函数，默认返回2000个特征以供下游分析如PCA分析
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
#plot1 + plot2


###---数据缩放---
#使用ScaleData()函数
#调整每个基因的表达水平，使所有细胞的平均表达水平为 0
#对每个基因的表达进行缩放，使细胞间的方差为 1，使下游分析中各基因权重相等，从而避免高表达基因占据主导地位
#结果存储在pbmc[["RNA"]]$scale.data
#默认情况下，只有可变特征会被缩放

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

#执行线性降维
#PCA
#对于第一主成分，Seurat 输出一个具有最多正载荷和负载荷的基因列表，代表在数据集中的单细胞中表现出相关性（或反相关性）的基因模块
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
#Examine and Visualize
#print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca") + NoLegend()
#最常用，便于探索数据集中异质性的主要来源，以决定哪些主成分需要纳入后续分析
#细胞和特征根据PCA得分进行排序，设置阈值cells，可加速绘制
DimHeatmap(pbmc,dims = 1, cells = 500, balanced = T)  #dims指的是数据维度，为多少就有几个小图
DimHeatmap(pbmc, dims = 1:15, cells = 500, balanced = T)

###---确定数据集的维度---
#采用生成“肘形图“进行分析
ElbowPlot(pbmc)
#可以观察到 PC9-10 附近存在一个“肘形”，表明大部分真实信号都包含在前 10 个主成分中
#提示：在选择此参数时尽量取较高值。例如，仅使用 5 个主成分进行下游分析会对结果产生显著的负面影响。

###------------细胞聚类-----------------
pbmc <- FindNeighbors(pbmc, dims = 1:10)   #此处dims是之前确定的
pbmc <- FindClusters(pbmc, resolution = 0.5)
#Tips：对于约 3000 个细胞的单细胞数据集，将分辨率设置为 0.4 到 1.2 之间通常可以获得良好的结果。对于更大的数据集，最佳分辨率通常会更高。
#head(Idents(pbmc),5)  #查看聚类结果

###----------非线性降维----------
pbmc <- RunUMAP(pbmc, dims = 1:10)
DimPlot(pbmc, reduction = "umap")    #图形形状相同即可，由于包自身随机数的选择可能出现中心对称翻转等情况



###----------寻找差异表达特征（聚类生物标志物）----------

#find all markers of cluster 2
cluster2.markers <- FindMarkers(pbmc, ident.1 = 2)
#head(cluster2.markers, 5)

cluster5.markers <- FindMarkers(pbmc, ident.1 = 5, ident.2 = c(0,3))
#head(cluster5.markers,5)
#将marker和细胞比较，只返回能比对上者
pbmc.markers <- FindAllMarkers(pbmc, only.pos = T)
pbmc.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)
#差异表达检测（ROC检验返回任何单个标记的分类能力，范围从 0 - 随机到 1 - 完美）
cluster0.marker <- FindMarkers(pbmc, ident.1 = 0, logfc.threshold = 0.25, test.use = "roc", only.pos = T)
#可视化标记表达
VlnPlot(pbmc, features = c("MS4A1","CD79A"))
VlnPlot(pbmc, features = c("NKG7","PF4"), slot = "counts", log = T)  #针对 raw counts
FeaturePlot(pbmc, features = c("MS4A1", "GNLY", "CD3E", "CD14", "FCER1A", "FCGR3A", "LYZ", "PPBP","CD8A"))

pbmc.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1) %>%
  slice_head(n = 10) %>%
  ungroup() -> top10
DoHeatmap(pbmc, features = top10$gene) + NoLegend()

#为每个细胞簇分配标识
new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T", "FCGR3A+ Mono", "NK", "DC", "Platelet")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, new.cluster.ids)
DimPlot(pbmc,reduction = "umap", label = T, pt.size = 0.5) + NoLegend()

library(Seurat)

#导出数据供MLP模型构建使用
# 假设对象为 pbmc
# 1. 导出 normalized counts（推荐模型输入）
norm_counts <- as.matrix(pbmc[["RNA"]]@data)
write.csv(norm_counts, file = "norm_counts.csv")

# 2. 导出 variable genes
var_genes <- VariableFeatures(pbmc)
write.csv(var_genes, file = "variable_genes.csv", row.names = FALSE)

# 3. 导出 PCA embeddings
pca_embeddings <- Embeddings(pbmc, "pca")
write.csv(pca_embeddings, file = "pca_embeddings.csv")

# 4. 导出 UMAP embeddings（用于可视化）
umap_embeddings <- Embeddings(pbmc, "umap")
write.csv(umap_embeddings, file = "umap_embeddings.csv")

# 5. 导出 cluster labels
clusters <- pbmc$seurat_clusters
write.csv(clusters, file = "clusters.csv")

# 6.（可选）导出 scaled data（更适合深度模型）
scaled_data <- pbmc[["RNA"]]@scale.data
write.csv(scaled_data, file = "scaled_data.csv")
