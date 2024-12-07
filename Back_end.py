from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
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
    # Configuration
    class Config:
        device = "cpu"
        thresholds_path = "/kaggle/working/Computer_vision/datasets/thresholds.pkl"
        detector_weight_path = "/kaggle/input/iumodel-dataset/IUmodel/diagnosisbot.pth"
        t_model_weight_path = "./weight_path/mimic_t_model.pth"
        image_size = 300
        dataset_name = 'iu_xray'
        theta = 0.4
        gamma = 0.4
        beta = 1.0
        mode = "infer"
        infer_path = "/kaggle/working/iu_xray_weight_epoch0_.pth"
        batch_size = 8

    config = Config()

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
if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)