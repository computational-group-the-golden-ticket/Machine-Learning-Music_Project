import torch
import torch.nn as nn

import multi_training


class BasicaModel(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(BasicaModel, self).__init__()

        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.n_layer = len(layer_sizes)

        last_output_size = self.input_size

        # Create the layers of the LSTM because only the input and output
        #   in stack can be specified.
        for i in range(self.n_layer):
            self.__setattr__('lstm%d' % i,
                             nn.LSTM(last_output_size, layer_sizes[i]))
            last_output_size = layer_sizes[i]

    def init_hidden(self, batch_size, to_cuda=False):
        hidden_states = []

        for i in range(self.n_layer):
            hidden_state = torch.randn(1, batch_size, self.layer_sizes[i])

            if to_cuda:
                hidden_state = hidden_state.cuda()

            hidden_states.append(hidden_state)

        return hidden_states

    def forward(self, x, to_cuda=False):
        # x.dim = time, batch_size, input
        self.hidden = self.init_hidden(x.shape[1], to_cuda)
        self.state = self.init_hidden(x.shape[1], to_cuda)

        last_output = x

        for i in range(self.n_layer):
            lstm = self.__getattr__('lstm%d' % i)

            last_output, _ = lstm(last_output, (self.hidden[i], self.state[i]))

        return last_output


class TimeModel(BasicaModel):
    def __init__(self, input_size, layer_sizes):
        super(TimeModel, self).__init__(input_size, layer_sizes)


class PitchModel(BasicaModel):
    def __init__(self, input_size, layer_sizes):
        super(PitchModel, self).__init__(input_size, layer_sizes)

        self.linear = nn.Linear(self.layer_sizes[self.n_layer - 1], 2)

    def forward(self, x, *args, **kwargs):
        last_output = super(PitchModel, self).forward(x, *args, **kwargs)

        last_output = torch.sigmoid(self.linear(last_output))

        return last_output


class Model(nn.Module):
    def __init__(self, t_layer_sizes, p_layer_sizes):
        super(Model, self).__init__()

        self.time_model = TimeModel(80, t_layer_sizes)

        self.pitch_model = PitchModel(self.time_model.layer_sizes[-1] + 2,
                                      p_layer_sizes)

    def forward(self, x):
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

    model = Model([300, 300], [100, 50])
    print(model)

    out = model(data)

    print(out.shape)


def main():
    test_time_model()
    test_pitch_model()
    test_model()


if __name__ == '__main__':
    main()
