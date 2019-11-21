import numpy as np
import torch

identifier = 'q_zero'

prediction_locs = {
    '1JHC': 'cormorant_1JHC_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '1JHN': 'cormorant_1JHN_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '2JHC': 'cormorant_2JHC_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '2JHN': 'cormorant_2JHN_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '2JHH': 'cormorant_2JHH_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '3JHC': 'cormorant_3JHC_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '3JHN': 'cormorant_3JHN_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier,
    '3JHH': 'cormorant_3JHH_continue_%s_ncg_5_top_mlp_maxl_3_nc_48/' % identifier}

kaggle_data = np.load('data/champs-scalar-coupling/targets_test_expanded.npz')
kaggle_data = {key: val for (key, val) in kaggle_data.items()}
all_jj_labels = np.hstack(kaggle_data['jj_label'])
predictions = np.zeros(len(all_jj_labels))

for label, loc in prediction_locs.items():
    predictions_i = torch.load(loc + "predict_out/nosave.best.test.pt")
    (mu, sigma) = predictions_i['stats']
    targets = predictions_i['predict'] * sigma + mu
    targets = targets.numpy().ravel()
    locs = np.where(all_jj_labels == label)[0]
    print('locs', locs[:4])
    print(targets[0])
    predictions[locs] = targets

predictions = predictions.reshape(-1, 1)
ids = np.arange(4658147, 7163688 + 1).reshape(-1, 1)
predictions = np.hstack((ids, predictions))
header = "id,scalar_coupling_constant"


np.savetxt('predictions_%s.csv' % identifier, predictions, header=header, delimiter=',', fmt=["%d", "%.4f"])
