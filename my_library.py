def test_load():
  return 'loaded'
  
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence_row, target, target_value):
  table_columns = up_list_column_names(table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)
  cond_prob_list = [cond_prob(table, e_col, e_val, target, target_value) for e_col, e_val in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  c_prob0 = cond_probs_product(table, evidence_row, target, 0)
  p_prob0 = prior_prob(table, target, 0)
  numerator0 = (c_prob0)*(p_prob0)
  c_prob1 = cond_probs_product(table, evidence_row, target, 1)
  p_prob1 = prior_prob(table, target, 1)
  numerator1 = (c_prob1)*(p_prob1)
  probs = compute_probs(numerator0, numerator1)
  return probs

def metrics(zipped_list):
  assert isinstance(zipped_list, list), f'zipped_list is not a list. It is {type(zipped_list)}'
  assert all([isinstance(col, list) for col in zipped_list]), f'zipped_list is not a list of lists.'
  assert all((isinstance(item, (list,tuple)) and len(item)==2) for item in zipped_list), f'zipped_list does not have each value as a pair of items'
for a,b in zipped_list:
   assert isinstance(a,(int,float)) and isinstance(b,(int,float)), f'zipped_list contains a non-int or non-float pair: {[a,b]}'
for a,b in zipped_list:
   assert float(a) in [0.0,1.0] and float(b) in [0.0,1.0], f'zipped_list contains a non-binary pair: {[a,b]}'  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])
  Precision = tp/(tp+fp) if (tp+fp) != 0 else 0
  Recall = tp/(tp+fn) if (tp+fn) != 0 else 0
  F1 = 2*((Precision*Recall)/(Precision+Recall)) if (Precision+Recall) != 0 else 0
  Accuracy = (tp + tn)/(tp + tn + fp + fn) if (tp + fp + tn + fn) != 0 else 0
  metrics_dict = {'Precision':tp/(tp+fp) if (tp+fp) != 0 else 0, 'Recall':tp/(tp+fn) if (tp+fn) != 0 else 0, 'F1':2*((Precision*Recall)/(Precision+Recall))if (Precision+Recall) != 0 else 0, 'Accuracy':(tp + tn)/(tp + tn + fp + fn) if (tp + fp + tn + fn) != 0 else 0}
  return metrics_dict

from sklearn.ensemble import RandomForestClassifier  
def run_random_forest(train, test, target, n):
  X = up_drop_column(train, target)
  y = up_get_column(train,target)  
  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)  
  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
  clf.fit(X, y)  
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]
  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]
  metrics_table = up_metrics_table(all_mets)
  return metrics_table

def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)
  for arch in architectures:
    all_results= up_neural_net(train_table, test_table, arch,target)
    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]
    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))
    return None 
