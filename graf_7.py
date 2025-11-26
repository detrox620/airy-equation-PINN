import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.special import airy  # для функции Эйри Ai(x)

from task_7 import PINN, a, b, layers, N_f


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Интервал и сетка для валидации
x_plot = np.linspace(a, b, N_f)
Ai_true, _, _, _ = airy(x_plot)  # airy(x) возвращает (Ai, Aip, Bi, Bip)

for activation in ['tanh', 'sigmoid', 'softplus', 'sin']:
    print(f"Activation function: {activation}")

    model = PINN(layers, activation=activation)
    model.load_state_dict(
        torch.load(f'models/model_weights_{activation}.pth', map_location=device)
    )
    model.eval()

    # Подготавливаем вход для модели
    x_f = torch.linspace(a, b, N_f).view(-1, 1)

    with torch.no_grad():
        y_pred = model(x_f).cpu().numpy().flatten()

    # Построение графика
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, Ai_true, 'k--', linewidth=1.5, label=r'Точное решение $\mathrm{Ai}(x)$')
    plt.plot(x_plot, y_pred, 'r', linewidth=1.2, label='PINN-аппроксимация')

    plt.xlabel('$x$')
    plt.ylabel('$y(x)$')
    plt.xlim(a, b)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.title(f'Решение уравнения Эйри: активация = {activation}')
    plt.legend()

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'grafs/displacement_plot_{activation}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Опционально: вычислить и напечатать MSE
    mse = np.mean((y_pred - Ai_true) ** 2)
    print(f"  MSE по сравнению с Ai(x): {mse:.2e}")
