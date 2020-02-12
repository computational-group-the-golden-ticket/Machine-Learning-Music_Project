# para borrar todo y empezar otra vez
%cd ..
!rm -rf /content/Machine-Learning-Music_Project/
%cd /content/

# cell 1
!ls

# cell 2
!git clone https://github.com/computational-group-the-golden-ticket/Machine-Learning-Music_Project.git
%cd Machine-Learning-Music_Project/

!git checkout luispapiernik-dev
!cat colab_code.py

# cell 3
from midi_to_statematrix import *

import multi_training
import model

from main import *
import os

# 5
import importlib as il

multi_training = il.reload(multi_training)
model = il.reload(model)

# cell 6
model_directory = "output"
os.makedirs(model_directory, exist_ok=True)
pcs = multi_training.loadPieces("Scale2")

start = get_last_epoch(model_directory)

m = model.BiaxialRNNModel([300, 300], [100, 50])
m.cuda()

multi_training.trainPiece(m, pcs, 10000, model_directory, start)
