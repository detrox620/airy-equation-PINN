import torch
import torch.nn as nn
from math import gamma
import time


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class PINN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self.activation_dict = nn.ModuleDict({
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'sin': Sin(),
        })
        if activation not in self.activation_dict:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = self.activation_dict[activation]

    # проход для каждой эпохи
    def forward(self, t):
        inputs = t
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        x = self.layers[-1](inputs)  # для последнего слоя не применяется активация
        return x


layers = [1, 150, 150, 150, 1]
a = -5.0
b = 3.0
N_f = 15000

y_0 = 1. / (3. ** (2. / 3.) * gamma(2. / 3.))
y_x_0 = -1. / (3. ** (1. / 3.) * gamma(1. / 3.))


def loss_function(model, x_f, x_i):
    y_f = model(x_f)
    y_i = model(x_i)

    # производные
    y_f_x = torch.autograd.grad(y_f, x_f,
                                grad_outputs=torch.ones_like(y_f),
                                create_graph=True)[0]
    y_f_xx = torch.autograd.grad(y_f_x, x_f,
                                 grad_outputs=torch.ones_like(y_f_x),
                                 create_graph=True)[0]

    # производная для н.у.
    y_i_x = torch.autograd.grad(y_i, x_i,
                                grad_outputs=torch.ones_like(y_i),
                                create_graph=True)[0]

    # уравнение
    f = y_f_xx - x_f * y_f

    loss_f = torch.mean(torch.squeeze(f) ** 2)
    loss_y0 = torch.mean((torch.squeeze(y_i) - y_0) ** 2)
    loss_y_x_0 = torch.mean((torch.squeeze(y_i_x) - y_x_0) ** 2)

    c1, c2, c3 = 100., 1., 1.
    loss = c1 * loss_f + c2 * loss_y0 + c3 * loss_y_x_0
    return loss, (loss_f.item(), loss_y0.item(), loss_y_x_0.item())


def learn(activation):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)

    model = PINN(layers, activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)

    x_f = torch.linspace(a, b, N_f, device=device).view(-1, 1).requires_grad_(True)
    x_i = torch.tensor(0., device=device).view(-1, 1).requires_grad_(True)

    num_epochs = 25000

    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        loss, losses = loss_function(model, x_f, x_i)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch},',
                  f'Loss_f = {round(losses[0], 4)},',
                  f'Loss_y0 = {round(losses[1], 4)},',
                  f'Loss_y_x_0 = {round(losses[2], 4)}')

    torch.save(model.state_dict(), f'models/model_weights_{activation}.pth')


if __name__ == '__main__':
    t = time.time()
    for activation in ['tanh', 'sigmoid', 'softplus', 'sin']:
        print(f"Обучение модели для функции активации {activation}:")
        learn(activation)
        print(f"Обучение модели для функции активации {activation} завершено.")
    print(f"Общее время обучения: {time.time() - t:.2f} секунд")
