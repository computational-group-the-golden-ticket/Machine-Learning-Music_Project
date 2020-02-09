# para borrar todo y empezar otra vez
!rm -rf /content/Machine-Learning-Music_Project
%cd /content/

# cell 1
!git clone https://github.com/computational-group-the-golden-ticket/Machine-Learning-Music_Project.git
%cd Machine-Learning-Music_Project/

!git checkout DEV-23
!cat colab_code.py

# cell 2
from midi_to_statematrix import *

import multi_training
import model

# cell 3
import importlib as il

multi_training = il.reload(multi_training)
model = il.reload(model)

# cell 4
pcs = multi_training.loadPieces("Train_data")

m = model.BiaxialRNNModel([300, 300], [100, 50])
m.cuda()

multi_training.trainPiece(m, pcs, 10000)
