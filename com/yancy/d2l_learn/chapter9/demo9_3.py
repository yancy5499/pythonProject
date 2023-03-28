import torch
from torch import nn
from d2l import torch as d2l
import com.yancy.d2l_learn.chapter8.demo8_5 as demo8_5

'''
深度循环神经网络(多层RNN)
'''

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 2.0
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers=2)  # 两层RNN
    model = d2l.RNNModel(lstm_layer, vocab_size)
    model.to(device)
    demo8_5.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
'''
困惑度 1.0, 115722.8 词元/秒 cuda:0
time traveller for so it will be convenient to speak of himwas e
travelleryou can show black is white by argument said filby
'''
