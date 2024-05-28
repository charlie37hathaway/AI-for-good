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
  assert all(all(isinstance(val, int) and val >=0 for val in item) for item in zipped_list), f'Arguments may not be negative and must be integers.'
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])
  Precision = tp/(tp+fp) if (tp+fp) != 0 else 0
  Recall = tp/(tp+fn) if (tp+fn) != 0 else 0
  F1 = 2*((Precision*Recall)/(Precision+Recall)) if (Precision+Recall) != 0 else 0
  Accuracy = (tp + tn)/(tp + tn + fp + fn) if (tp + fp + tn + fn) != 0 else 0
  metrics_dict = {'Precision':tp/(tp+fp) if (tp+fp) != 0 else 0, 'Recall':tp/(tp+fn) if (tp+fn) != 0 else 0, 'F1':2*((Precision*Recall)/(Precision+Recall))if (Precision+Recall) != 0 else 0, 'Accuracy':(tp + tn)/(tp + tn + fp + fn) if (tp + fp + tn + fn) != 0 else 0}
  return metrics_dict
