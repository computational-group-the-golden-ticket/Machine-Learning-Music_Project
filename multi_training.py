import os
import random
from midi_to_statematrix import *
from data import *
import numpy
import signal

import torch

# Number of examples to be sampled in the minimization.
batch_width = 10
# Length of each sequence; the batch are desired to contain 8 bars.
batch_len = 16 * 8
# Interval between possible start locations; this is the number of notes that
#   can be resolved in a bar.
division_len = 16


def loadPieces(dirpath):

    # Make a dicitonary to load all pieces
    pieces = {}

    # For over the file in  driectory to verify if they are midi files
    for fname in os.listdir(dirpath):
        # I not midi file try with the next
        if fname[-4:] not in ('.mid', '.MID'):
            continue

        # Save name without .mid termination
        name = fname[:-4]

        # Transform the midi song to data as indicated in
        #   midiToNoteStateMatrix in midi_to_statematrix.py
        outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        # If this is not the case, it is upload because it has several possible
        #   examples
        if len(outMatrix) < batch_len:
            continue
        pieces[name] = outMatrix

        # Load info
        print("Loaded {}".format(name))

    return pieces


def getPieceSegment(pieces):
    # Choose a piece uploaded; all the pieces are assign with the same
    #   probability.
    piece_output = random.choice(list(pieces.values()))
    # Choose a inintial position in a random bar.
    start = random.randrange(0, len(piece_output) - batch_len, division_len)
    # Take a segment of batch_len size; this represent batch_len times
    #   the duration of the shortest possible note.
    seg_out = piece_output[start:start + batch_len]
    # This will change the last dimension of seg_out in order to give to each
    #  note context about the notes played and hold in the surrounding times
    #  as well as other parameters.
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out


def getPieceBatch(pieces):
    # The input and oputut are copied in a a tuple, each batch_width times in
    #   the following way i = (seg_in, ..., seg_in) and o = (seg_out, ....,
    #   seg_out).
    i, o = zip(*[getPieceSegment(pieces) for _ in range(batch_width)])
    return torch.Tensor(i), torch.Tensor(o)


def NLLLoss(source, target):
    loss = 2 * source * target - source - target + 1 + 1e-14
    loss = -torch.sum(torch.log(loss))

    return loss


def train(model, pieces):
    # This implemnes the Negative log likelihood function in order to be
    #  minimized; "sum" option sums all the components of the tensor into a
    #  scalar.
    # loss_function = torch.nn.MSELoss(reduction='sum')
    # loss_function = torch.nn.NLLLoss(reduction='sum')
    loss_function = torch.nn.BCELoss(reduction='sum')
    # loss_function = torch.nn.functional.nll_loss
    # loss_function = NLLLoss.apply

    # The problem adapt well to the benefits that are refered in (Adam et. al.,
    #   2015). Between them it is the fact that is good for non-stationary
    #   as will be expected for this type of problem, in which changing a note
    #   can produce models with the same quality.
    optimizer = torch.optim.Adam(model.parameters())

    # Get a part of the song, for more details see the function.
    input_mat, output_mat = getPieceBatch(pieces)
    input_mat = input_mat.cuda()
    output_mat = output_mat.cuda()

    # Run forward model for the data
    output = model((input_mat, output_mat), training=True)

    active_notes = torch.unsqueeze(output_mat[:, 1:, :, 0], dim=3)
    mask = torch.cat([torch.ones_like(active_notes, device='cuda'), active_notes], dim=3)

    output = mask * output

    output_mat = output_mat[:, 1:]

    output = torch.unsqueeze(output.reshape((-1, 1)), dim=1)
    output_mat = torch.unsqueeze(output_mat.reshape((-1, 1)), dim=1)

    # Calculate NLLLoss, gradients and actualizing parameters; the numbers are
    #   pass to long, this ony works for long. Prediction is passed first,
    #   expected probabilities are passed as second parameter.
    output = output.reshape(output.shape[0])
    output_mat = output_mat.reshape(output.shape[0])
    loss = loss_function(output, output_mat)
    # loss = loss_function(output.long(), output_mat.long())

    loss.backward()
    optimizer.step()

    return loss


def trainPiece(model, pieces, epochs, start=0):
    stopflag = [False]

    def signal_handler(signame, sf):
        stopflag[0] = True

    old_handler = signal.signal(signal.SIGINT, signal_handler)

    for i in range(start, start + epochs):
        if stopflag[0]:
            break

        # Making the training for each epoch
        error = train(model, pieces)

        # Each 100 epochs print the error
        if i % 100 == 0:
            print("epoch {}, error={}".format(i, error))

        # This saves the model each 100 if less than 1000 epochs and 500 epochs
        #   after this
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            # This Choose the seed for the predcition, and to make a
            #   predicition to see how the net is doing.
            xIpt, xOpt = map(torch.Tensor, getPieceSegment(pieces))

            # noteStateMatrixTomidi expect numpy array inputs
            init_notes = numpy.expand_dims(xOpt[0].numpy(), axis=0)
            seed_tensor = xIpt[0].cuda()
            # Input is tensor, int
            predict_notes = model(seed_tensor, batch_len)
            predict_notes = numpy.array(predict_notes)

            cpu_tensor = predict_notes.cpu()

            dummy_notes = (init_notes, cpu_tensor)
            noteStateMatrixTomidi(numpy.concatenate(dummy_notes, axis=0),
                                  'output/sample{}'.format(i))

            # Save the model
            torch.save(model.state_dict(), 'output/params{}.p'.format(i))

    signal.signal(signal.SIGINT, old_handler)
