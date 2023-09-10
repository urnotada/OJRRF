import pickle
import torch
def get_emb_dict():
        with open("./datasets/job/concatenated_file.pkl", "rb") as tf:
            emb_dict = pickle.load(tf)
        return emb_dict

def get_emb(user_ids,device):
        emb_dict = get_emb_dict()
        embs = []
        for i in user_ids:

            embs.append(emb_dict[i])
        embs = torch.tensor(embs, dtype=torch.float32,device=device)
        # print(embs)
        return embs