import torch
from model.layer.nalu import NaluLayer


def make_data(batch_size=50, MIN=1, MAX=100, func=lambda a, b: a + b):
    x = torch.randint(MIN, MAX, (batch_size, 2, ))
    y = func(x[..., 0], x[..., 1]).unsqueeze(1)
    return x, y


model = NaluLayer(2, 1, 2, 4)
op = torch.optim.Adam(model.parameters())
func = lambda a, b: a.sqrt() * b
#func = lambda a, b: a + b

loss_sum = 0.
for i in range(100000):
    x, y = make_data(func=func)
    y_ = model(x)

    loss = (y - y_).pow(2).mean()
    op.zero_grad()
    loss.backward()
    op.step()

    if i % 1000 == 0:
        loss_sum /= 1000
        print(i, loss_sum)
        if float(loss_sum) < 0.01:
            print(x[0], y[0], y_[0])
    loss_sum += float(loss)


x, y = make_data(func=func, MIN=101, MAX=1000)
y_ = model(x)

loss = (y - y_).abs().mean()
print('E', loss)
print(x[0], y[0], y_[0])
