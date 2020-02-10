# Import standart libraries for ML
import torch
import torch.nn as nn
import numpy as np

from data import noteStateSingleToInputForm


class BasicModel(nn.Module):
    """
    This is a class to represent a recurrent stack neuronal network
    with layers of arbitrary sizes.

    input_size(int): size of input layer.
    layer_sizes(list): contain the sizes for the remaning layers.

    e.g: input_size = 30 and layer_sizes = [20, 40, 10] creates a NN with
    layers of 30, 20, 40, and 10.
    """

    def __init__(self, input_size, layer_sizes):
        super(BasicModel, self).__init__()

        # Making layer sizes attributes of the class.
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        # Parameter to make a for and construct the remaining layers
        self.n_layer = len(layer_sizes)

        last_output_size = self.input_size

        # Create the layers of the LSTM one by one, this is not done with
        #   stack long short term memory nn package because this one only
        #   permits same size for the middle layers; which is not the case
        #   for works as this one; note that for each we give a different name.
        for i in range(self.n_layer):
            self.__setattr__('lstm%d' % i,
                             nn.LSTM(last_output_size, layer_sizes[i]))
            last_output_size = layer_sizes[i]

    def init_hidden(self, batch_size, to_cuda=False):
        """
        This routine initialize the the hiddden states of all the stacked
        layers, remember each stack layers communicate with itself in the next
        recurrent step an with the other in the same recurrent step.

        batch_size(int): number of examples.
        to_cuda(bool): indication to make the objects in cuda in order to
          paralelize
        """
        hidden_states = []

        for i in range(self.n_layer):
            # Init made for each hidden layer according to N(0,1), in oder
            #  for better backprop behaviour.
            # The first parameter is one beacuse the stack initialization is
            #  made for each latyer indiviaully.
            hidden_state = torch.randn(1, batch_size, self.layer_sizes[i],
                                       requires_grad=True)

            # If cuda paralelization option is on, then use it.
            if to_cuda:
                hidden_state = hidden_state.cuda()

            # Saving all the initial hidden states
            hidden_states.append(hidden_state)

        return hidden_states

    def forward(self, x, to_cuda=False):
        """
        This method implements the general forward for a stack lstm neuronal
          network; with the different layers already defined.
        """
        # Remmber that: x.dim = (n_seq, batch_size, input_size). In standart
        #  notation the next lines are writen h and c respectively.
        self.hidden = self.init_hidden(x.shape[1], to_cuda)
        self.state = self.init_hidden(x.shape[1], to_cuda)

        # Init the the last output variable to make explicit calculations
        last_output = x

        # Do forward calculations passing through every lstm cell; thanks to
        #  the nn library each lstm1, ..., lstmN connect to itself in the other
        #  step; the gradient descent will be well mada thanks to the fact that
        #  calculations process is save to then apply autograd.
        for i in range(self.n_layer):
            lstm = self.__getattr__('lstm%d' % i)

            last_output, _ = lstm(last_output, (self.hidden[i], self.state[i]))

        return last_output


class TimeModel(BasicModel):
    """
    This is a clear copy of BasicModel and it is done to make everyting more
      clear for the Music_generation project.
    """
    pass


class PitchModel(BasicModel):
    """
    This is almost a copy of BasicModel, in addition it makes the following:
      1. An additional layer is add to the lstm with two units in order to
        modelate this as the probability of the input note to be hold or
        articulated; the input note is not given derectly to PitchModel, this
        will be clearer in BiaxialRNNModel.
    """

    def __init__(self, input_size, layer_sizes):
        super(PitchModel, self).__init__(input_size, layer_sizes)

        # This is the two layer size that was mentioned.
        self.linear = nn.Linear(self.layer_sizes[-1], 2)

    def probabilities2notes(self, x):
        threshold = np.random.rand()

        shouldPlay = threshold < x[0]
        shouldArtic = shouldPlay * (threshold < x[1])

        return torch.Tensor([shouldPlay, shouldArtic])

    def forward(self, x, *args, **kwargs):
        # This call the forward specified in BasicModel.
        last_output = super(PitchModel, self).forward(x, *args, **kwargs)
        # This apply the last 2 dim layer.
        last_output = torch.sigmoid(self.linear(last_output))

        return last_output


class BiaxialRNNModel(nn.Module):
    """
    This is designed to contain the implmentation inspired in the repo
      https://github.com/hexahedria/biaxial-rnn-music-composition and its
      respective blog where is explained with some degree of detail.
    The folloing lines try to summary the process meade with the data during
      the network.

    1. Data that was extracted from midi file songs in a tensorial form is
      given as input, for more details see the commented code midi_to_state_ma-
      trix.py.
    2. The algorithm runs a stack LSTM RNN over time dimension after collapsing
      the note  dimensions and batch_size into one; this net is desired to
      discover all possible time connections between the data in an ouput
      hidden_state.
    3. After this, the hidden_state vector of features is resize in order to
      contain additional info and data is reshaped in order to run a stacked
      lstm RNN over the notes; time and batch_size are collapsed in one
      dimension.
      Also in the each note step a two dimensional ouput is modelated in order
      to simulate the probability of the input note (in 1. above) to be hold
      and articulated; note that thanks to the fact that as this depend of
      having the correct ouput on the last one, this oput probabilities are
      really conditional ones; and so the cost function is moedelated as the
      log likelihood. For more details about this see https://www.youtube.com/
      watch?v=FHRQ2DhQjRM&list=PLZnyIsit9AM7yeTZuBmezKNc6hFHUPImh&index=6
    """

    def __init__(self, t_layer_sizes, p_layer_sizes):
        super(BiaxialRNNModel, self).__init__()

        # This init the time model; for more info about why 80, see http://www.
        #   hexahedria.com/2015/08/03/composing-music-with-recurrent-neural
        #   -networks/
        self.time_model = TimeModel(80, t_layer_sizes)

        # The dimension of the last layer size is increase by two, to include
        #  information of the last note in each note step, this one says if
        #  the last note was hold and played.
        self.pitch_model = PitchModel(self.time_model.layer_sizes[-1] + 2,
                                      p_layer_sizes)

    def local_train(self, x):
        """
        This forward is basicaly the implmentation on detail of what was said
          in te description of the class.

        x(torch.Tensor): Input information of the notes played in different
          songs, note that x.dim = ... complemented
        """

        # output_mat is defined by the function midiToNoteStateMatrix in the
        #  file midi_to_state_matrix; while input_mat has the last dimension
        #  changed, with other information that talk about the pitch, real
        #  frequency of the note, its sorrounding notes, the part that it
        #  occupies in the bar and other info.
        input_mat, output_mat = x

        # Check how exactly this here affect the code
        # input_mat.requires_grad = True
        # output_mat.requires_grad = True

        # The last info in time dimension is eliminated, as each data in the
        #   matrix (time, batch, number_notes, features) try to predic the next
        #   one, that is to say, time=0 predicts time=1, then, time=1 predicts
        #   time=2... and timed=n_time-2 predicts time=n_time-1; output that
        #   is used later to compare with the knwon notes in order to train the
        #   NN.
        input_slice = input_mat[:, 0:-1]
        # Update dimensions.
        n_batch, n_time, n_note, n_ipn = input_slice.shape

        # Permute and collapse the dimensions with n_time first to run the
        #    RNN to find ouputs that contain the time connections.
        time_inputs = input_slice.permute((1, 0, 2, 3))
        time_inputs = time_inputs.reshape((n_time, n_batch * n_note, n_ipn))

        # Run time_model
        last_output = self.time_model(time_inputs)

        # Get the n_hidden; number of the features that contain the time
        #  connections.
        n_hidden = self.time_model.layer_sizes[-1]

        # Collapse data in order to run a RNN over the notes.
        last_output = last_output.reshape(
            (n_time, n_batch, n_note, n_hidden)).permute(
            (2, 1, 0, 3)).reshape((n_note, n_batch * n_time, n_hidden))

        # The last note try to modelate the next, and for the first note
        #  the only thing that we have is silence, then a silence note
        #  is added.
        start_note_values = torch.zeros(1, last_output.shape[1], 2,
                                        requires_grad=False)

        # This is the matrix that represents the notes to be predicted. The
        #   last dim in note is erased again thanks to the fact that that this
        #   does not predict a next one that can be compared; the firs time is
        #   also erased because there is not first time note to predcit it, in
        #   order to compare.
        correct_choices = output_mat[:, 1:, 0:-1, :].permute(
            (2, 0, 1, 3)).reshape((n_note - 1, n_batch * n_time, 2))

        # Concate the silence that we have mentioned.
        note_choices_inputs = torch.cat([start_note_values, correct_choices],
                                        dim=0)
        # We add to  the hidden feature that contain the time information
        #  another two components that say if taht note was played in the
        #  last one.
        note_inputs = torch.cat([last_output, note_choices_inputs], dim=2)

        # Apply the stacked LTSM RNN trough note sequence.
        last_output = self.pitch_model(note_inputs)

        # Save the expectation of the hold and play probabilities for the
        #   notes.
        last_output = last_output.reshape(
            (n_note, n_batch, n_time, 2)).permute(1, 2, 0, 3)

        return last_output

    def predict_one_step(self, x):
        input_mat = torch.unsqueeze(x, dim=0)

        hidden_time = self.time_model(input_mat)

        last_note_values = torch.Tensor([0, 0])
        next_notes_step = []  # list to append the new now generated
        for i in range(hidden_time.shape[1]):
            note_input = torch.cat([hidden_time[0][i], last_note_values])
            note_input = torch.unsqueeze(note_input, dim=0)
            note_input = torch.unsqueeze(note_input, dim=0)

            probabilities = self.pitch_model(note_input)
            last_note_values = self.pitch_model.probabilities2notes(probabilities[0][0])

            next_notes_step.append([last_note_values[0].long().item(),
                                    last_note_values[1].long().item()])

        return next_notes_step

    def predict_n_steps(self, x, n):
        note_state_matrix = []

        last_step = x
        for i in range(n):
            last_step = self.predict_one_step(last_step)

            note_state_matrix.append(last_step)

            last_step = noteStateSingleToInputForm(last_step, i)
            last_step = torch.Tensor(last_step)

        return note_state_matrix

    def forward(self, x, n=1, training=False):
        if training:
            return self.local_train(x)
        else:
            return self.predict_n_steps(x, n=n)


def main():
    pass


if __name__ == '__main__':
    main()
