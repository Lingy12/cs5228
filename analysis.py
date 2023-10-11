import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_r_2(df, x, y, transform = lambda x: x):
    df = df.copy()
    df[x] = df[x].map(transform)
    df[[x,y]] = MinMaxScaler().fit_transform(df[[x,y]])
    x = df[x].values
    y = df[y].values
    y_mean = np.mean(y)
    y_pred = np.poly1d(np.polyfit(x, y, 1))(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return r2