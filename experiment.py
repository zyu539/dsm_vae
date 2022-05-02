import numpy as np
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
from sklearn.model_selection import ParameterGrid
from auton_survival import datasets
from auton_survival.preprocessing import Preprocessor
from model import DSMVAE

# Load Dataset
outcomes, features = datasets.load_support()
cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp',
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph',
             'glucose', 'bun', 'urine', 'adlp', 'adls']

features = Preprocessor().fit_transform(features, cat_feats=cat_feats, num_feats=num_feats)

# Initialize timeline
horizons = [0.25, 0.5, 0.75]
times = np.quantile(outcomes.time[outcomes.event == 1], horizons).tolist()

# Grid Search Params settings
param_grid = {'n': [3, 4, 5],
              'k': [3, 4, 5],
              'learning_rate': [1e-4, 1e-3],
              'layers': [[50, 100], [100, 100]],
              'discount': [0.3, 0.5, 1.0]
              }
params = ParameterGrid(param_grid)

# Get training, val, test dataset ready
x, t, e = features.values, outcomes.time.values, outcomes.event.values

n = len(x)

tr_size = int(n * 0.70)
vl_size = int(n * 0.10)
te_size = int(n * 0.20)

x_train, x_test, x_val = x[:tr_size], x[-te_size:], x[tr_size:tr_size + vl_size]
t_train, t_test, t_val = t[:tr_size], t[-te_size:], t[tr_size:tr_size + vl_size]
e_train, e_test, e_val = e[:tr_size], e[-te_size:], e[tr_size:tr_size + vl_size]

# Tune hyper parameters
models = []
for param in params:
    model = DSMVAE(k=param['k'],
                   n=param['n'],
                   layers=param['layers'],
                   discount=param['discount'])
    # The fit method is called to train the model
    model.fit(x_train, t_train, e_train, iters=100, learning_rate=param['learning_rate'])
    models.append([[model.compute_nll(x_val, t_val, e_val), model]])
best_model = min(models)
model = best_model[0][1]

# predict

out_risk = model.predict_risk(x_test, times)
out_survival = model.predict_survival(x_test, times)

# experiments
cis = []
brs = []

et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                    dtype=[('e', bool), ('t', float)])
et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                   dtype=[('e', bool), ('t', float)])
et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                  dtype=[('e', bool), ('t', float)])

for i, _ in enumerate(times):
    cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])

brs.append(brier_score(et_train, et_test, out_survival, times)[1])
roc_auc = []

for i, _ in enumerate(times):
    roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])

for horizon in enumerate(horizons):
    print(f"For {horizon[1]} quantile,")
    print("TD Concordance Index:", cis[horizon[0]])
    print("Brier Score:", brs[0][horizon[0]])
    print("ROC AUC ", roc_auc[horizon[0]][0], "\n")
