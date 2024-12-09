from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import argparse
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from models import utils, caption
from datasets import xray
from utils.engine import evaluate
from models.model import swin_tiny_patch4_window7_224 as create_model
from utils.stloss import SoftTarget

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Streamlit frontend
origins = ["*"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_methods=[""], allow_headers=[""]
)

def build_diagnosisbot(num_classes, detector_weight_path):
    model = create_model(num_classes=num_classes)
    model.load_state_dict(torch.load(detector_weight_path, map_location=torch.device('cpu')), strict=True)
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model

@app.post("/evaluate")
async def evaluate_xray(data_dir: str = Form(...), anno_path: str = Form(...)):

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr_drop', type=int, default=20)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Backbone
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--dilation', type=bool, default=True)
    # Basic
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--clip_max_norm', type=float, default=0.1)

    # Transformer
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vocab_size', type=int, default=760)
    parser.add_argument('--start_token', type=int, default=1)
    parser.add_argument('--end_token', type=int, default=2)

    parser.add_argument('--enc_layers', type=int, default=6)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--pre_norm', type=int, default=True)

    # diagnosisbot
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--thresholds_path', type=str, default="datasets/thresholds.pkl")
    parser.add_argument('--detector_weight_path', type=str, default="H:/CV_project/IUmodel/diagnosisbot.pth")
    parser.add_argument('--t_model_weight_path', type=str, default="./weight_path/mimic_t_model.pth")
    parser.add_argument('--knowledge_prompt_path', type=str, default="H:/CV_project/IUmodel/knowledge_prompt_iu.pkl")

    # ADA
    parser.add_argument('--theta', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=1.0)

    # Delta
    parser.add_argument('--delta', type=float, default=0.01)

    # Dataset
    parser.add_argument('--image_size', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='iu_xray')
  
    parser.add_argument('--limit', type=int, default=1)

    # mode
    parser.add_argument('--mode', type=str, default="infer")
    parser.add_argument('--test_path', type=str, default="H:/CV_project/IUmodel/iu_xray_weight_epoch0_.pth")

    parser.add_argument('--infer_path', type=str, default="H:/CV_project/IUmodel/iu_xray_weight_epoch0_.pth")
    # parser.add_argument('--infer_limit', type=int, default=1)


    config = parser.parse_args()


    # Initialize models
    device = torch.device(config.device)
    detector = build_diagnosisbot(14, config.detector_weight_path)
    detector.to(device)

    model, criterion = caption.build_model(config)
    model.to(device)

    # Load thresholds
    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    # Prepare dataset
    dataset_test = xray.build_dataset(
        config, mode='test', anno_path=anno_path, data_dir=data_dir,
        dataset_name=config.dataset_name, image_size=config.image_size,
        theta=config.theta, gamma=config.gamma, beta=config.beta
    )
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, config.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=dataset_test.collate_fn)

    # Evaluate
    if config.mode == "infer":
        if os.path.exists(config.infer_path):
            weights_dict = torch.load(config.infer_path, map_location='cpu')['model']
            model.load_state_dict(weights_dict, strict=False)

        test_result = evaluate(
            model, detector, criterion, data_loader_test, device, config,
            thresholds=thresholds, tokenizer=dataset_test.tokenizer, mode=config.mode
        )
        return JSONResponse(content={"report": test_result})

# Run the backend
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)