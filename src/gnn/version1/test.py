import torch
dtf  = torch.load(r'F:\FORTH_Final_Thesis\FORTH-Thesis\gnn\version1\dataset_dtf\train_graphs.pt',  weights_only=False)
pdc  = torch.load(r'F:\FORTH_Final_Thesis\FORTH-Thesis\gnn\version1\dataset_pdc\train_graphs.pt',  weights_only=False)

print(dtf[0].edge_attr[:5])
print(pdc[0].edge_attr[:5])
print(dtf[0].x[:3])
print(pdc[0].x[:3])
