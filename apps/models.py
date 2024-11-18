import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###

        self.model = nn.Sequential(
          self.ConvBN(3,16,7,4,device = device,dtype=dtype),
          self.ConvBN(16,32,3,2,device = device,dtype = dtype),
          nn.Residual(
            nn.Sequential(
              self.ConvBN(32,32,3,1,device = device,dtype= dtype),
              self.ConvBN(32,32,3,1,device = device,dtype= dtype)
            )

          ),
          self.ConvBN(32,64,3,2,device = device,dtype= dtype),
          self.ConvBN(64,128,3,2,device = device,dtype= dtype),
          nn.Residual(
            nn.Sequential(
              self.ConvBN(128, 128, 3, 1,device = device,dtype = dtype),
              self.ConvBN(128, 128, 3, 1, device = device,dtype= dtype)

            )
          ),
          nn.Flatten(),
          nn.Linear(128,128,device = device,dtype = dtype),
          nn.ReLU(),
          nn.Linear(128,10,device = device,dtype = dtype),


        )
    def ConvBN(self, a, b, k, s,device,dtype):
        return nn.Sequential(
            nn.Conv(a, b, kernel_size = k, stride = s, device = device, dtype = dtype),
            nn.BatchNorm2d(b, device = device, dtype = dtype),
            nn.ReLU()
        )


        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.model(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        self.hidden_size = hidden_size
        # # Sequence model (RNN or LSTM)
        if seq_model == 'rnn':
            self.seq_model_layer = nn.RNN(embedding_size, hidden_size, 
                                        num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model_layer = nn.LSTM(embedding_size, hidden_size, 
                                         num_layers=num_layers, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown sequence model: {seq_model}")
        
        # Final linear layer
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        # Get embeddings
        # x: (seq_len, bs) -> embedded: (seq_len, bs, embedding_size)
        embedded = self.embedding(x)
        
        # Pass through sequence model
        # output: (seq_len, bs, hidden_size)
        # h: shape depends on model type (RNN vs LSTM)
        output, h = self.seq_model_layer(embedded, h)
        
        # Reshape output for linear layer
        seq_len, bs, _ = output.shape
        
        output = output.reshape((seq_len * bs, self.hidden_size))
        
        # Pass through linear layer to get logits
        # output: (seq_len*bs, output_size)
        output = self.linear(output)
        
        return output, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)