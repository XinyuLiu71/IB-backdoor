import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_info_plane():
    inputs_outputs_arr = np.load('results/ob_infoNCE_06_22/infoNCE_MI_log_inputs_vs_outputs.npy'
                                 , allow_pickle=True)
    Y_outputs_arr = np.load('results/ob_infoNCE_06_22/infoNCE_MI_log_Y_vs_outputs.npy'
                            , allow_pickle=True)
    print(inputs_outputs_arr.shape, Y_outputs_arr.shape)  # (11, ) (11, 100)
    info_plane = np.empty([11, 2])
    for idx in range(11):
        info_plane[idx, 0] = np.mean(inputs_outputs_arr[idx][-1])
        info_plane[idx, 1] = np.mean(Y_outputs_arr[idx][-1])
    # x轴: inputs, Y轴: outputs
    df = pd.DataFrame(columns=['Epoch', 'I(X;T)', 'I(T;Y)'])
    df['Epoch'] = np.arange(0, 101, 10)
    df['I(X;T)'] = info_plane[:, 0]
    df['I(T;Y)'] = info_plane[:, 1]
    fig, ax = plt.subplots()
    sca = ax.scatter(x=df['I(X;T)'], y=df['I(T;Y)'], c=df['Epoch'], cmap='summer')
    ax.set_xlabel('I(X;T)')
    ax.set_ylabel('I(T;Y)')
    fig.colorbar(sca, label="Epoch", orientation="vertical")
    fig.show()


def test():
    arr = np.load(r'figs-inputs-vs-outputs/infoNCE.npy')
    print(arr.shape)
    pass


if __name__ == '__main__':
    plot_info_plane()
    # test()
