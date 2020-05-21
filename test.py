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
    correct_predictions = labels.int().eq(reference.int())
    return correct_predictions.sum().float() / correct_predictions.nelement()

def train(epochs, batch_size, lr, verbose):

    # autograd globally off
    torch.set_grad_enabled(False)
    # generate training and testing datasets
    train_data, train_label = generate_data()
    test_data, test_label = generate_data()
    # normalize data be centered at 0
    train_data, test_data = normalize(train_data, test_data)

    if verbose:
        print("--- Dataset ---")
        print("Train X: ", train_data.size(), " | Train y: ", train_label.size())
        print(" Test X: ",  test_data.size(), " |  Test y: ", test_label.size())

    layers =[]
    # input layer (2 input units)
    linear1 = Linear(2, 25, bias= True, weight_init=xavier_uniform)

    # 3 hidden layers (each 25 units)
    linear2 = Linear(25, 25, bias= True, weight_init=xavier_uniform)
    linear3 = Linear(25, 25, bias= True, weight_init=xavier_uniform)
    linear4 = Linear(25, 25, bias= True, weight_init=xavier_uniform)

    # output layer (2 output units)
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
    if verbose: print("Number of model parameters: {}".format(sum([len(p) for p in model.param()])))

    criterion = MSE()
    optimizer = SGD(model, lr=lr)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    train_errors, test_errors = [], []

    if verbose: print("--- Training ---")
    for epoch in range(1, epochs+1):
        if verbose:print("Epoch: {}".format(epoch))

        # TRAINING
        for batch_idx in range(0, train_data.size(0), batch_size):
            # axis 0, start from batch_idx until batch_idx+batch_size
            output = model.forward(train_data.narrow(0, batch_idx, batch_size))

            # Calculate loss
            loss = criterion.forward(output, train_label.narrow(0, batch_idx, batch_size))
            train_losses.append(loss)
            if verbose: print("Train Loss: {:.2e}".format(loss.item()))

            # put to zero weights and bias
            optimizer.zero_grad()

            ## Backpropagation
            # Calculate grad of loss
            loss_grad = criterion.backward()

            # Grad of the model
            model.backward(loss_grad)

            # Update parameters
            optimizer.step()

        train_prediction = model.forward(train_data)
        acc = accuracy(train_prediction, train_label)
        train_accuracies.append(acc)
        train_errors.append(1-acc)
        if verbose: print("Train Accuracy: {:.2e}".format(acc.item()))

        # EVALUATION
        for batch_idx in range(0, test_data.size(0), batch_size):
            # axis 0, start from batch_idx until batch_idx+batch_size
            output = model.forward(test_data.narrow(0, batch_idx, batch_size))

            # Calculate loss
            loss = criterion.forward(output, test_label.narrow(0, batch_idx, batch_size))
            test_losses.append(loss)
            if verbose: print("Test Loss: {:.2e}".format(loss.item()))

            # put to zero weights and bias
            #optimizer.zero_grad()

            ## Backpropagation
            # Calculate grad of loss
            #loss_grad = criterion.backward()

            # Grad of the model
            #model.backward(loss_grad)

            # Update parameters
            #optimizer.step()

        test_prediction = model.forward(test_data)
        acc = accuracy(test_prediction, test_label)
        test_accuracies.append(acc)    
        test_errors.append(1-acc)
        if verbose: print("Test Accuracy: {:.2e}".format(acc.item()))

    return train_losses, test_losses, train_accuracies, test_accuracies, train_errors, test_errors 
        

if __name__ == "__main__":
    epochs = 100
    batch_size = 100
    verbose = True
    lr = 0.1

    train(epochs, batch_size, lr, verbose);
    