from typing import List
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.models.unet import UNet
from src.data.datasets import ACDCDataset


@st.cache(allow_output_mutation=True)
def load(tagged: bool = True) -> Dataset:
    dataset: Dataset = ACDCDataset(tagged=tagged)
    return dataset


st.sidebar.title('Performance evaluator')

model_store: List[Path] = Path('checkpoints/model').iterdir()
model_path: str = st.sidebar.selectbox('Select model', model_store)

model: nn.Module = UNet(n_channels=1, n_classes=4, bilinear=True).double()
saved_model: nn.Module = torch.load(model_path)
if isinstance(saved_model, nn.DataParallel):
    saved_model = saved_model.module
model.load_state_dict(saved_model.state_dict())
model.to('cpu')

tagged: bool = st.sidebar.radio('Image style', ('Tagged', 'Cine')) == 'Tagged'
dataset = load(tagged=tagged)

img: int = st.slider('Select datapoint', value=1, min_value=0, max_value=len(dataset), step=1)

inp: torch.Tensor = dataset[img][0].unsqueeze(0).double().clone()
out: torch.Tensor = model(inp)
out = F.softmax(out, dim=1).argmax(dim=1).detach().numpy()[0]

fig, axes = plt.subplots(1, 3)

axes[0].imshow(dataset[img][0][0], cmap='gray')
axes[1].imshow(dataset[img][1])
axes[2].imshow(out)

for ax in axes:
    ax.axis('off')

st.pyplot(fig)
