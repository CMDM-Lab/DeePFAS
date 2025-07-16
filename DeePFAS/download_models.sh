
ae_model=https://zenodo.org/records/15083140/files/ae_best_model.pt
deepfas_with_over_model=https://zenodo.org/records/15083140/files/deepfas_r2_over_best_model.pt
deepfas_r1_with_over_model=https://zenodo.org/records/15083140/files/deepfas_r1_over_best_model.pt
deepfas_without_over_model=https://zenodo.org/records/15083140/files/deepfas_r2_best_model.pt

wget -O ae/ae_saved/ae_best_model.pt $ae_model
wget -O DeePFAS/deepfas_saved/deepfas_r2_over_best_model.pt  $deepfas_with_over_model
wget -O DeePFAS/deepfas_saved/deepfas_r2_best_model.pt  $deepfas_without_over_model
wget -O DeePFAS/deepfas_saved/deepfas_r1_over_best_model.pt $deepfas_r1_with_over_model