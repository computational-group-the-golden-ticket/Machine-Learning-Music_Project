# Impor modules for training
import torch
import model as m
import multi_training


def test_time_model():
    time_model = m.TimeModel(80, [300, 300])
    print(time_model)

    data = torch.randn(127, 780, 80)

    out = time_model(data)

    print(out.shape)


def test_pitch_model():
    pitch_model = m.PitchModel(80, [300, 300])
    print(pitch_model)

    data = torch.randn(127, 780, 80)

    out = pitch_model(data)

    print(out.shape)


def test_training_model():
    pcs = multi_training.loadPieces("Train_data")
    data = multi_training.getPieceBatch(pcs)

    model = m.BiaxialRNNModel([300, 300], [100, 50])
    print(model)

    out = model(data, training=True)

    print(out.shape)


def test_predict_one_step_model():
    pcs = multi_training.loadPieces("Train_data")
    data, _ = multi_training.getPieceSegment(pcs)
    data = torch.Tensor(data[0])
    print(data.shape)

    model = m.BiaxialRNNModel([300, 300], [100, 50])
    print(model)

    out = model(data)

    print(out)


def test_predict_n_step_model():
    pcs = multi_training.loadPieces("Train_data")
    data, _ = multi_training.getPieceSegment(pcs)
    data = torch.Tensor(data[0])
    print(data.shape)

    model = m.BiaxialRNNModel([300, 300], [100, 50])
    print(model)

    out = model(data, 5)

    for i in out:
        print(i)


def main():
    # test_time_model()
    # test_pitch_model()
    # test_training_model()
    # test_predict_one_step_model()
    test_predict_n_step_model()


if __name__ == '__main__':
    main()
