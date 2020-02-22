# Machine-Learning-Music_Project

This one is desired to be the next host of an algorithm capable of composing new creative songs based in training data songs. To begin we have followed the first week of the course of "Deep Learning Specialization: Sequence models" in which composition of Musica Jazz was obtained with success (see my_music10000.midi). Now, with this understaning of seq2seq models we proceed to a more complicated and complete model given in http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/ for which we first get an understanding about the midi files and the extaction of data as is illustrated in midi_to_state_matrix.py, later we use tensorflow library in order to take advantage of the cuda GPU's offered by colab and the connection that can be made with Google Drive to save the output and restart the run at any time, and used the structure defined in the cited post.


