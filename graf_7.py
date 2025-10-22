import torch
import matplotlib.pyplot as plt
import time
import pandas as pd

from task_7 import PINN, a, b, layers, N_f

model = PINN(layers)

for activation in ['tanh', 'sigmoid', 'softplus', 'sin']:
    print(f"Activation function: {activation}")
    model.load_state_dict(torch.load(f'models/model_weights_{activation}.pth', weights_only=True))
    model.eval()

    x_f = torch.linspace(a, b, N_f).view(-1, 1)

    with torch.no_grad():
        y_f = model(x_f)

    approx_data = pd.DataFrame(dict(
        x=x_f.numpy().flatten(),
        y=y_f.detach().numpy().flatten(),
    ))

    axis = approx_data.plot(
        y="y",
        x="x",
        color="tab:red",
        label="Решение уравнения Эйри"
    )
    axis.set_ylabel("$y$")
    axis.set_xlabel("$x$")
    axis.set_xlim(a, b)
    axis.grid(True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'grafs/displacement_plot_{activation}.png', dpi=300, bbox_inches='tight')
    plt.show()
