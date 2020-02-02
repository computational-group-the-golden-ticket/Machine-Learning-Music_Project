import os
import random
from midi_to_statematrix import *
from data import *

import signal

import torch

batch_width = 10  # number of sequences in a batch
batch_len = 16 * 8  # length of each sequence
division_len = 16  # interval between possible start locations


def loadPieces(dirpath):

    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid', '.MID'):
            continue

        name = fname[:-4]

        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        if len(outMatrix) < batch_len:
            continue

        pieces[name] = outMatrix
        print("Loaded {}".format(name))

    return pieces


def getPieceSegment(pieces):
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, len(piece_output) - batch_len, division_len)

    seg_out = piece_output[start:start + batch_len]
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out


def getPieceBatch(pieces):
    i, o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return torch.Tensor(i), torch.Tensor(o)


def train(model, pieces):
    loss_function = torch.nn.NLLLoss(reduction='sum')
    # optimizer = torch.optim.Adam(model.parameters())

    input_mat, output_mat = getPieceBatch(pieces)

    output = model((input_mat.cuda(), output_mat.cuda()))

    print(output_mat[:, 1:].shape, output.shape)
    loss = loss_function(output.long().cuda(), output_mat[:, 1:].long().cuda())

    loss.backward()
    # optimizer.step()

    return loss


def trainPiece(model, pieces, epochs, start=0):
    stopflag = [False]

    def signal_handler(signame, sf):
        stopflag[0] = True

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    for i in range(start, start + epochs):
        if stopflag[0]:
            break

        error = train(model, pieces)
        error = 0
        if i % 100 == 0:
            print("epoch {}, error={}".format(i, error))

        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xIpt, xOpt = map(torch.Tensor, getPieceSegment(pieces))

            # noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(
            #     batch_len, 1, xIpt[0])), axis=0), 'output/sample{}'.format(i))

            torch.save(model.state_dict(), 'output/params{}.p'.format(i))

    signal.signal(signal.SIGINT, old_handler)
