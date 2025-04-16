import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

# 采集样本点
h = 10 # 网格密度
N_i = 10 # 内点采集个数
N_b = 5 # 边界点采集个数
N_data = 10 # 数据点采集个数
def inter(n=N_i):
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    cond = (2 - x**2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def down_yy(n=N_b):
    x = torch.rand(n, 1)
    y = torch.zeros_like(x)
    cond = x**2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def up_yy(n=N_b):
    x = torch.rand(n, 1)
    y = torch.ones_like(x)
    cond = x**2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def down(n=N_b):
    x = torch.rand(n, 1)
    y = torch.zeros_like(x)
    cond = x**2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def up(n=N_b):
    x = torch.rand(n, 1)
    y = torch.ones_like(x)
    cond = x**2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def left(n=N_b):
    y = torch.rand(n, 1)
    x = torch.zeros_like(y)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def right(n=N_b):
    y = torch.rand(n, 1)
    x = torch.ones_like(y)
    cond = torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def data_inter(n=N_data):
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    cond = x**2 * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond


# 定义梯度计算和PDE损失
def grad(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return grad(grad(u, x), x, order=order-1)

loss = nn.MSELoss()
def l_inter(u):
    x, y, cond = inter()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(grad(u_xy, x, 2) - grad(u_xy, y, 4), cond)

def l_down_yy(u):
    x, y, cond = down_yy()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(grad(u_xy, y, 2), cond)

def l_up_yy(u):
    x, y, cond = up_yy()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(grad(u_xy, y, 2), cond)

def l_down(u):
    x, y, cond = down()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(u_xy, cond)

def l_up(u):
    x, y, cond = up()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(u_xy, cond)

def l_left(u):
    x, y, cond = left()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(u_xy, cond)

def l_right(u):
    x, y, cond = right()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(u_xy, cond)

def l_data(u):
    x, y, cond = data_inter()
    u_xy = u(torch.cat([x, y], dim=1))
    return loss(u_xy, cond)


# 建立模型
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


# 训练
epochs = 300
u = net()
opt = torch.optim.Adam(params=u.parameters(), lr=0.01)
log = []
for epoch in range(epochs):
    opt.zero_grad()
    l_all = l_inter(u)+l_up_yy(u)+l_down_yy(u)+l_up(u)+l_down(u)+l_left(u)+l_right(u)+l_data(u)
    l_all.backward()
    opt.step()
    if epoch % 10 == 0:
        log.append([epoch, l_all.detach()])
        print(f"epoch: {epoch}, loss: {l_all}")


# Inference 产生0-1之间的网格点用于test
xc = torch.linspace(0, 1, h)
xm, ym = torch.meshgrid(xc, xc)  # 100*100个点的平面中，xm表示所有点的x坐标，ym表示y坐标
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_pred = u(xy)
u_real = xx * xx * torch.exp(-yy)
u_error = torch.abs(u_pred - u_real)
u_pred_fig = u_pred.reshape(h, h)
u_real_fig = u_real.reshape(h, h)
u_error_fig = u_error.reshape(h, h)
# print("Max abs error is: ", float(torch.max(torch.abs(u_pred - xx * xx * torch.exp(-yy)))))
print("Max abs error is: ", float(torch.max(torch.abs(u_pred - u_real))))
# 仅有PDE损失    Max abs error:  0.004852950572967529
# 带有数据点损失  Max abs error:  0.0018916130065917969

# loss曲线
log = np.array(log)
fig = plt.figure(figsize=(14, 6))
ax_2d = fig.add_subplot(121)
ax_2d.plot(log[:, 0], log[:, 1])
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()
# fig.savefig("Loss curve.png")

# 真解和pinn解的散点图
ax_3d = fig.add_subplot(122, projection='3d')
ax_3d.scatter(xx.detach().numpy(), yy.detach().numpy(), u_pred.detach().numpy())
ax_3d.scatter(xx.detach().numpy(), yy.detach().numpy(), u_real.detach().numpy())
plt.legend(['pinn', 'real'])
plt.show()
fig.savefig("Loss.png")

# 作PINN数值解图
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig.detach().numpy())
ax1.text2D(0.5, 0.9, "PINN solve", transform=ax1.transAxes)
# plt.show()
fig.savefig("PINN solve.png")

# 作真解图
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_real_fig.detach().numpy())
ax2.text2D(0.5, 0.9, "real solve", transform=ax2.transAxes)
# plt.show()
fig.savefig("real solve.png")

# 误差图
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_error_fig.detach().numpy())
ax3.text2D(0.5, 0.9, "abs error", transform=ax3.transAxes)
plt.show()
fig.savefig("abs error.png")
