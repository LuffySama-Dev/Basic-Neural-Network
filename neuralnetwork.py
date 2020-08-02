# import numpy for random arrays etc
import numpy
# import scipy for activation funciton and etc
import scipy.special
# Neural Network Class Definition
class neuralNetwork:
    # Initialise the neural Network
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):
        # Set the number of node in each input, output, and hidden layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #learning Rate
        self.lr = learningrate

        # Link weight matrices, wih and who
        # Weights inside the arrays are w_i_j, where link id from node i to node j in the next layer #w11 and w22 etc
        #self.wih = (numpy.random.rand(self.hnodes, self.inodes)-0.5)
        #self.who = (numpy.random.ramd(self.onodes, self.hnodes)-0.5)

        # More Sophisticated weights
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # Activation fucntion is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        # We use lambda to create a function there and then, quickly and easily. THe function takes x as input and returns sigmoid function.
        # Whenevr someone needs to use the activation function they just need to call the self.activation_function().


    # Training the network is divided into two parts.
    # 1] The first part is working out the output for a given training example. That is no different to what we just did with the query() funciton.
    # 2] The second part is taking this calculated output, comparing it with the desired output, and using the difference to guide the updating of the network weights.
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Claculate signals into hidden layer.
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #Calculate signal into outputlayer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Output layer error
        output_error = targets-final_outputs

        # Hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_error)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_error * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self,inputs_list):
        # convert input list into 2d
        inputs = numpy.array(inputs_list,ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # Calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into the final layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # Calculate signals emerging from finaly layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# Number of input, hidden and output input_nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.1
learning_rate = 0.1

# Create instance of neural networks
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Load the mnist training data csv file into a input_list
training_data_file = open("D:\Learning\Machine Learning Program\mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Train the neural networks
epochs = 5
for e in range(epochs):
    # Go through all the records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99) + 0.01
        targets = numpy.zeros(output_nodes) +0.1
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# Load the mnist test data CSV file into a input_list
test_data_file = open("D:\Learning\Machine Learning Program\mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural networks
# Scorecard for how well the network performs, initially empty
scorecard = []

# Go through all the records in the test dat set
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy,asfarray(all_values[1:])/255.0*0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

scorecard_array = numpy.asfarray(Scorecard)
print("Performance = ", scorecard_array.sum()/scorecard_array.size)
