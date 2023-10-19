import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats


def plot_info_plane():
    inputs_outputs_arr = np.load('results/ob_infoNCE_06_22/infoNCE_MI_log_inputs_vs_outputs.npy'
                                 , allow_pickle=True)
    Y_outputs_arr = np.load('results/ob_infoNCE_06_22/infoNCE_MI_log_Y_vs_outputs.npy'
                            , allow_pickle=True)
    info_plane = np.empty([49, 2])
    for idx in range(49):
        info_plane[idx, 0] = np.mean(inputs_outputs_arr_5_clean[idx])
        info_plane[idx, 1] = np.mean(Y_outputs_arr_clean010[idx])
    # x轴: inputs, Y轴: outputs
    df = pd.DataFrame(columns=['Epoch', 'I(X;T)', 'I(T;Y)'])
    df['Epoch'] = np.arange(0, 98, 2)
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


def manovaAnalyze():
    # 读取数据
    data = pd.read_csv('poison_clean.csv')
    print(data.dtypes)

    exog = data[['x']]
    endog = data[['intial_x', 'initial_y', 'turning_x', 'turning_y', 'conv_x', 'conv_y']].astype(float)

    # 进行MANOVA分析
    maov = MANOVA(endog, exog)
    print(maov.mv_test())


def annovaAnalyze():
    # Annova
    data = pd.read_csv('annova.csv')
    print(data.dtypes)
    # annova
    f_statistic, p_value = stats.f_oneway(data['intial_x'], data['initial_y'], data['turning_x'], data['turning_y'], data['conv_x'], data['conv_y'])

    print('F statistic:', f_statistic)
    print('P value:', p_value)

if __name__ == '__main__':
    plot_info_plane()
    # test()
    # manovaAnalyze()
    # annovaAnalyze()
