Hello, myself Saurabh Patil learning about Machine Learning and the neural networks. While searching for a book which will guide me through ML and Neural networks, I came across Make your own neural network book. It is written by Tariq Rashid.

It is divided into three parts. Firstly writer makes us go through the theory and trust me it is interesting. The way writer has explained everything, any noob like me can understand it very easily.

While writing this readme.txt, I have already went through Part one (Theory).
I am in middle of part two ( DIY with python ). All the code in DIY with python can be found in my GitHub repo.

So, until now I have learned how the neural network works.

Now it's time to build one.

As we all know everything needs a skeleton, architecture or support or a frame. So, we build a Frame or Skeleton Code according to writer.

If you will read the theory you will know that there are three important functions for a neural network to work.
1] Initialization - to set the number of input, hidden and output node.
2] Train - refine the weights after being give a training set example to learn from.
3] Query - Give an answer from the output nodes after being given an input.

Once we have created the skeleton, we set the number of nodes in each layer.
Then we create matrix for linking weights of nodes
Then we create queries which will take input to the neural network and returns output.
To link the weights we use dot product.
once we are done with the query, we will move forward to training.
Training the network is divided into two parts.
  1] The first part is working out the output for a given training example. That is no different to what we just did with the query() function.
  2] The second part is taking this calculated output, comparing it with the desired output, and using the difference to guide the updating of the network weights.

We will use the MINIST dataset of handwritten Numbers.

I like the way writer has explained everything. I am uploading the same code as in book for reference. If anyone has trouble with it email me, I will remove it.

You can see the code in neuralNetwork.py


Thanks and Regard
Saurabh Patil
