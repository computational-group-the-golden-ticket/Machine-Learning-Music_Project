# Import standart libraries for ML
import torch
import torch.nn as nn

# Impor modules for training
import multi_training


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

        # Making layer sizes attributes of the class
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        # Parameter to make a for and construct the remaining layers
        self.n_layer = len(layer_sizes)

        last_output_size = self.input_size

        # Create the layers of the LSTM one by one, this is not done with
        #   stack long short term memory nn package because this one only
        #   permits same size for the middle layers; which is not the case
        #   for other works; note that for each we give a different name.
        for i in range(self.n_layer):
            self.__setattr__('lstm%d' % i,
                             nn.LSTM(last_output_size, layer_sizes[i]))
            last_output_size = layer_sizes[i]

    def init_hidden(self, batch_size, to_cuda=False):
        """
        This routine initialize the the hiddden states of all the stacked
        layers, remember stack laters communicate with one another.

        batch_size(int): number of examples.
        to_cuda(bool): indication to make the objects in cuda in order to
          paralelize
        """
        hidden_states = []

        for i in range(self.n_layer):
            # Init made for each hidden layer according to N(0,1), in oder
            #  for better backprop behaviour, the first parameter is one
            #  beacuse stacked layers are not desired to have the same dim.
            hidden_state = torch.randn(1, batch_size, self.layer_sizes[i])

            # If cuda paralelization option is on, then use it.
            if to_cuda:
                hidden_state = hidden_state.cuda()

            # Saving all the initial hidden states
            hidden_states.append(hidden_state)

        return hidden_states

    def forward(self, x, to_cuda=False):
        """
        This method implements the general forward for a stack lstm neuronal
          network with the different layers already defined.
        """
        # Remmber that: x.dim = (n_seq, batch_size, input_size). In standart
        #  notation they are writen h and c respectively.
        self.hidden = self.init_hidden(x.shape[1], to_cuda)
        self.state = self.init_hidden(x.shape[1], to_cuda)

        # Init the the last output variable to make explicit calculations
        last_output = x

        # Do forward calculations passing through every lstm cell; thanks to
        #  the nn library each lstm1, ..., lstmN connect to itself in the other
        #  step.
        for i in range(self.n_layer):
            lstm = self.__getattr__('lstm%d' % i)

            last_output, _ = lstm(last_output, (self.hidden[i], self.state[i]))

        return last_output


class TimeModel(BasicModel):
    """
    This is a clear copy of BasicModel and it is done to make everyting more
      clear for the Music_generation project.
    """

    def __init__(self, input_size, layer_sizes):
        super(TimeModel, self).__init__(input_size, layer_sizes)


class PitchModel(BasicModel):
    """
    This is almost a copy of BasicModel, in addition it makes the following:
      1. An additional layer is add to the lstm with two units in order to
        modelate this as the probability of the input note to be hold or
        articulated; the input note is given to TimeModel, this will be clearer
        in Model.
    """

    def __init__(self, input_size, layer_sizes):
        super(PitchModel, self).__init__(input_size, layer_sizes)

        # This is the two layer size that was mentioned.
        self.linear = nn.Linear(self.layer_sizes[-1], 2)

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
    The folloing lines try to summary the process meake with the data during
      the network.

    1. Data that was extracted from midi file songs in a tensorial form is
      given as input, for more details see the commented code midi_to_state_ma-
      trix.py.
    2. The algorithm runs a RNN over time dimension after collapsing the note
      dimensions and batch_size into one; this net is desired to discover all
      possible time connections between the data in an ouput hidden_state.
    3. After this, the hidden_state vector of features is resize in order to
      contain additional info and data is reshaped in order to run a stacked
      lstm RNN over the notes; time and batch_size are collapsed in one
      dimension.
      Also in the each note step a two dimensional ouput is modelated in order
      to simulate the porbability of the input note (in 1) to be hold and
      articulated; note that thanks to the fact that as this depend of having
      the correct ouput on the last one, this oput probabilities are really
      conditional ones; and so the cost function is moedelated as the log
      likelihood. For more details about this see https://www.youtube.com/watch
      ?v=FHRQ2DhQjRM&list=PLZnyIsit9AM7yeTZuBmezKNc6hFHUPImh&index=6
    """

    def __init__(self, t_layer_sizes, p_layer_sizes):
        super(BiaxialRNNModel, self).__init__()

        # This init the time model; for more info about why 80, see http://www.
        #   hexahedria.com/2015/08/03/composing-music-with-recurrent-neural
        #   -networks/
        self.time_model = TimeModel(80, t_layer_sizes)

        # The dimension of the last layer size is increase by two, to include
        #  information of the last note in each note step, this one says if
        #  the last note was hold and played
        self.pitch_model = PitchModel(self.time_model.layer_sizes[-1] + 2,
                                      p_layer_sizes)

    def forward(self, x):
        """
        This forward is basicaly the implmentation on detail of what was said
          in te description of the class.
        x(numpy_array): Input information of the notes played in different
          songs, note that x.dim = ... complemented
        """
        input_mat, output_mat = x

        input_slice = input_mat[:, 0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.shape

        time_inputs = input_slice.permute((1, 0, 2, 3))
        time_inputs = time_inputs.reshape((n_time, n_batch * n_note, n_ipn))

        last_output = self.time_model(time_inputs)

        n_hidden = self.time_model.layer_sizes[-1]

        last_output = last_output.reshape(
            (n_time, n_batch, n_note, n_hidden)).permute(
            (2, 1, 0, 3)).reshape((n_note, n_batch * n_time, n_hidden))

        start_note_values = torch.zeros(1, last_output.shape[1], 2)

        correct_choices = output_mat[:, 1:, 0:-1, :].permute(
            (2, 0, 1, 3)).reshape((n_note - 1, n_batch * n_time, 2))

        note_choices_inputs = torch.cat([start_note_values, correct_choices],
                                        dim=0)

        note_inputs = torch.cat([last_output, note_choices_inputs], dim=2)

        last_output = self.pitch_model(note_inputs)

        last_output = last_output.reshape(
            (n_note, n_batch, n_time, 2)).permute(1, 2, 0, 3)

        return last_output


def test_time_model():
    time_model = TimeModel(80, [300, 300])
    print(time_model)

    data = torch.randn(127, 780, 80)

    out = time_model(data)

    print(out.shape)


def test_pitch_model():
    pitch_model = PitchModel(80, [300, 300])
    print(pitch_model)

    data = torch.randn(127, 780, 80)

    out = pitch_model(data)

    print(out.shape)


def test_model():
    pcs = multi_training.loadPieces("music")
    data = multi_training.getPieceBatch(pcs)

    model = BiaxialRNNModel([300, 300], [100, 50])
    print(model)

    out = model(data)

    print(out.shape)


def main():
    test_time_model()
    test_pitch_model()
    test_model()


if __name__ == '__main__':
    main()
