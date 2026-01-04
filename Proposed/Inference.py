import torch

from AhNet import CFG, AhNet_MLP, get_data, predict

# ====================== Config ======================
""" Lorenz, Rossler, 
Chen, Chua, HyperLorenz, 
RabinovichFabrikant """

SYSTEM_NAME = "Chua"
SAVE_DIR = "./BestModel"
MODEL_PATH = f"{SAVE_DIR}/{SYSTEM_NAME}.pth"
DEVICE = CFG["device"]


def load_and_evaluate():

    _, X_test, _, _, scaler, data_scaled, feat_dim = get_data()
    model = AhNet_MLP(feat_dim, CFG["look_back"]).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    predict(model, X_test, scaler, data_scaled, feat_dim)


if __name__ == "__main__":

    load_and_evaluate()