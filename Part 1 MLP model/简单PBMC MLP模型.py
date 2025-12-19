import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

def safe_plot_xy(x, y, filename, title="Plot"):
    w, h = 800, 800
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    # 标题
    draw.text((10, 10), title, fill="black")

    # 归一化
    if len(x) == 0: return
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    eps = 1e-9
    x_norm = (x - x_min) / (x_max - x_min + eps)
    y_norm = (y - y_min) / (y_max - y_min + eps)

    # 绘制点
    for i in range(len(x_norm)):
        px = int(x_norm[i] * (w - 40) + 20)
        py = int((1 - y_norm[i]) * (h - 40) + 20)
        draw.ellipse((px-2, py-2, px+2, py+2), fill="blue")

    img.save(filename)
    print(f"[SafePlot] 已生成图片：{filename}")


def safe_plot_curve(curve1, curve2, filename, label1="Train", label2="Test"):
    """
    绘制 loss 曲线（保证输出 PNG）
    """
    w, h = 800, 600
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    draw.text((10, 10), f"{label1}/{label2} Loss Curve", fill="black")

    # 归一化
    all_vals = np.array(curve1 + curve2)
    mn, mx = all_vals.min(), all_vals.max()
    eps = 1e-9

    def norm(v):
        return (np.array(v) - mn) / (mx - mn + eps)

    c1 = norm(curve1)
    c2 = norm(curve2)

    # 画曲线
    for i in range(1, len(c1)):
        x1 = int((i-1)/len(c1) * (w-40) + 20)
        y1 = int((1-c1[i-1]) * (h-40) + 20)
        x2 = int(i/len(c1) * (w-40) + 20)
        y2 = int((1-c1[i]) * (h-40) + 20)
        draw.line((x1,y1,x2,y2), fill="red", width=2)

    for i in range(1, len(c2)):
        x1 = int((i-1)/len(c2) * (w-40) + 20)
        y1 = int((1-c2[i-1]) * (h-40) + 20)
        x2 = int(i/len(c2) * (w-40) + 20)
        y2 = int((1-c2[i]) * (h-40) + 20)
        draw.line((x1,y1,x2,y2), fill="blue", width=2)

    img.save(filename)
    print(f"[SafePlot] 已生成图片：{filename}")


# ================================
# 载入数据
# ================================
print(">>> Loading PCA/UMAP ...")
pca = pd.read_csv("D:/临时文件/AIVC/简单PBMC分析/pca_embeddings.csv", index_col=0)
umap = pd.read_csv("D:/临时文件/AIVC/简单PBMC分析/umap_embeddings.csv", index_col=0)

X = torch.tensor(pca.values, dtype=torch.float32)
Y = torch.tensor(umap.values, dtype=torch.float32)

dataset = TensorDataset(X, Y)
train_size = int(0.85*len(dataset))
train_ds, test_ds = random_split(dataset, [train_size, len(dataset)-train_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=64)


# ================================
# 高精度 MLP 模型
# ================================
class BigMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)


model = BigMLP(X.shape[1], Y.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_fn = nn.MSELoss()

# ================================
# 训练
# ================================
train_losses = []
test_losses = []

print(">>> Start training...")
for epoch in range(200):
    model.train()
    total = 0
    for bx, by in train_loader:
        opt.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        opt.step()
        total += loss.item()
    train_losses.append(total/len(train_loader))

    # test
    model.eval()
    total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx)
            loss = loss_fn(pred, by)
            total += loss.item()
    test_losses.append(total/len(test_loader))

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/200 | Train={train_losses[-1]:.4f} | Test={test_losses[-1]:.4f}")


# ================================
# 保存 Loss 曲线（SafePlot）
# ================================
safe_plot_curve(train_losses, test_losses, "training_curve.png")


# ================================
# 生成 UMAP 测试集预测（SafePlot）
# ================================
model.eval()
with torch.no_grad():
    X_test = X[test_ds.indices]
    Y_test = Y[test_ds.indices]
    Y_pred = model(X_test).numpy()

Y_test = Y_test.numpy()

safe_plot_xy(Y_test[:,0], Y_test[:,1], "umap_real.png", title="Real UMAP")
safe_plot_xy(Y_pred[:,0], Y_pred[:,1], "umap_pred.png", title="Predicted UMAP")

print(">>> 全部图片已成功生成！")
