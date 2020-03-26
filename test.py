import torch

from Sequential import Sequential
from activations.ReLu import Relu
from activations.Tanh import Tanh
from helpers.data_generator import generate_data, normalize
from layers.Linear import Linear
from losses.mse import MSE
from optimizers.SGD import SGD
from weight_initialization.xavier_uniform import xavier_uniform


def accuracy(predicted_logits, reference, argmax=True):
    """Compute the ratio of correctly predicted labels"""
    if argmax:
        labels = torch.argmax(predicted_logits, 1)
    else:
        labels = predicted_logits
    correct_predictions = labels.eq(reference)
    return correct_predictions.sum().float() / correct_predictions.nelement()

torch.set_grad_enabled(False)

train_data, train_label = generate_data()
test_data, test_label = generate_data()

train_data, test_data = normalize(train_data, test_data)

layers =[]
#input layer
linear1 = Linear(2, 25, bias= True, weight_init=xavier_uniform)

#hidden layers
linear2 = Linear(25, 25, bias= True, weight_init=xavier_uniform)
linear3 = Linear(25, 25, bias= True, weight_init=xavier_uniform)
linear4 = Linear(25, 25, bias= True, weight_init=xavier_uniform)

#output layer
linear5 = Linear(25, 2, bias= True, weight_init=xavier_uniform)


layers.append(linear1)
layers.append(Relu())
layers.append(linear2)
layers.append(Relu())
layers.append(linear3)
layers.append(Relu())
layers.append(linear4)
layers.append(Tanh())
layers.append(linear5)

model = Sequential(layers)

criterion = MSE()
optimizer = SGD(model, lr = 0.01)

for e in range(50):
    for b in range(0, train_data.size(0), 100):

        output = model.forward(train_data.narrow(0, b, 100))

        # Calculate loss
        loss = criterion.forward(output, train_label.narrow(0, b, 100))

        # put to zero weights and bias
        optimizer.zero_grad()

        ##Backpropagation
        # Calculate grad of loss
        loss_grad = criterion.backward()

        # Grad of the model
        model.backward(loss_grad)

        # Update parameters
        optimizer.step()

        #print(loss)

    test_prediction = model.forward(test_data)
    print(accuracy(test_prediction, test_label))