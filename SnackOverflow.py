import numpy as np
from sklearn.linear_model import LinearRegression

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)

    # Linear regression
    predictions = np.zeros(nins)
    for i in range(nins):
        X = np.arange(max(nt-35, 0), nt).reshape(-1, 1)
        y = prcSoFar[i, max(nt-35, 0):nt]
        if len(y) < 2:
            predictions[i] = prcSoFar[i, -1]
        else:
            model = LinearRegression()
            model.fit(X, y)
            next_time_step = np.array([[nt]])
            predictions[i] = model.predict(next_time_step)[0]

    lastRet = np.log(predictions / prcSoFar[:, -1])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm

    rpos = np.array([int(5000 * x / prcSoFar[i, -1]) for i, x in enumerate(lastRet)])

    buy_positions = np.array([x if x > 0 else 0 for x in rpos])
    sell_positions = np.array([x if x < 0 else 0 for x in rpos])

    currentPos = np.array([int(x) for x in currentPos + buy_positions + sell_positions])

    return currentPos