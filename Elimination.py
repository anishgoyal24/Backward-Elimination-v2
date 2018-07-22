def elimination(x, y, sl):
    import numpy as np
    import statsmodels.api as sm
    import pandas as pd

    column = []
    x_ots = pd.DataFrame(np.append(np.ones((x.shape[0], 1)).astype(int), x, axis=1))
    ols_predict = sm.OLS(exog=x_ots, endog=y).fit()
    r_value = float("{0:.4f}".format(ols_predict.rsquared_adj))
    x_ots_column = x_ots.shape[1]
    for i in range(1, x_ots_column):
        temp = pd.DataFrame(x_ots)
        x_ots = x_ots.drop(x_ots.columns[i], axis=1)
        ols_predict = sm.OLS(exog=x_ots, endog=y).fit()
        r = float("{0:.4f}".format(ols_predict.rsquared_adj))
        if r_value > r:
            x_ots = temp
        elif r_value <= r:
            column.append(int(i))
            r_value = r
            x_ots = temp
    return p_elimination(x_ots.drop(x_ots.columns[column], axis=1), y, sl)

def p_elimination(x, y, sl):
    x = pd.DataFrame(x)
    noCols = x.shape[1]
    for i in range(0, noCols):
        ols_regeressor = sm.OLS(exog=x, endog=y).fit()
        pValues = ols_regeressor.pvalues
        if max(pValues) > sl:
            x = x.drop(np.argmax(pValues), axis=1)
        else:
            break
    return x
