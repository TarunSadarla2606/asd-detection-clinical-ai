from pipeline import run_subject_pipeline
from model import load_model

m = load_model('weights/xai_cnn_best_weights.pth')
r = run_subject_pipeline('demo_subjects/32016.nii', m)

print('Pipeline OK:', r['subject_pred'], r['subject_conf'])