import pandas as pd
from model import load_model
from pipeline import run_subject_pipeline

df = pd.read_csv('5320_ABIDE_Phenotypics_20230908.csv', skiprows=1)
df['anon_num'] = df['ID'].str.replace('A','',regex=False).astype(int, errors='ignore')

model = load_model('weights/xai_cnn_best_weights.pth')

subjects = [32016, 32067, 32152, 32164, 32477, 32495, 33379, 33476, 33480]

print(f"{'ID':<8} {'True':>6} {'Pred':>6} {'Conf':>8} {'Match':>6}")
print("-" * 40)

for anon in subjects:
    row = df[df['anon_num'] == anon]
    true_label = {1:'ASD', 2:'TC'}.get(int(row['ABIDE_02'].values[0]), '?') if len(row) else '?'
    
    result = run_subject_pipeline(f'demo_subjects/{anon}.nii', model, top_k=3)
    pred   = result['subject_pred']
    conf   = result['subject_conf']
    match  = '✓' if pred == true_label else '✗'
    
    print(f"{anon:<8} {true_label:>6} {pred:>6} {conf:>8.1%} {match:>6}")