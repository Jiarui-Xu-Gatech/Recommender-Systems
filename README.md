# Recommender-Systems

# Data Description: MovieLens

MovieLens dataset (ml-latest-small) contains 100836 ratings and 3683 tag applications across 9742 movies. In this homework, we offer two files train.csv and test.csv, which will be used for training and testing the model, respectively. Each line of these files represents one rating of one movie by one user, and has the following format:
userId,movieId,rating,timestamp

# Item-Item Collaborative Filtering
The first recommendation model we are developing is a collaborative filtering algorithm using an item-oriented approach.

First, we use the cosine similarity to compute similarity between items  π  and  π  as follows.

π ππ(π,π)=πΈππ‘πππβπΈππ‘πππ/||πΈππ‘πππ||2||πΈππ‘πππ||2 

where  πΈππ‘πππ  is the pre-trained embedding of item  π ,  β  denotes an inner product between vectors, and  ||π||2  is a L2-norm of a vector  π .

With the above similarity function, we can compute the predicted rating  ππ’,π  on an item  π  for a user  π’  by computing the weighted sum of the ratings given by the user on the other items except  π .
ππ’,π=ββπβ π,(π’,π)βπ(π ππ(π,π)ππ’,π)/ββπβ π,(π’,π)βπ(|π ππ(π,π)|)

# Matrix Factorization

The second recommendation algorithm is called matrix factorization. Specifically, the matrix factorization algorithm works by decomposing the rating matrix into the product of two lower dimensionality rectangular matrices. Letβs define  πΎ  as the number of latent factors (or reduced dimensionality). Then, we learn a user profile  πββπΓπΎ  and an item profile  πββπΓπΎ  ( π  and  π  are the number of users and items, respectively). We want to approximate a rating by an inner product of two length- πΎ  vectors, one representing user profile and the other item profile. Mathematically, a rating  ππ’,π  of the user  π’  on the movie  π  is approximated by
ππ’,πββπ=1πΎππ’,πΒ·ππ,π

# Deep learning-based Recommendation

![image](https://github.com/Jiarui-Xu-Gatech/Recommender-Systems/blob/main/NN.png)

As shown in the above figure, the MLP model consists of embedding layer, hidden layer (or neural CF layers in the figure), and output layer.

The embedding layer transforms user_id and item_id to user and item latent vectors, respectively. Although the weights of the embedding layer are usually trainable, we will use pre-trained embeddings of users and items for faster learning speed.

The hidden layer consists of multiple fully connected (FC) layers with different sizes. Each FC layer 1) takes previous layer's output as input, 2) multiply them with its weight, and 3) applies an activation function like ReLU, Sigmoid, etc. (also bias term), and 4) send the output to the next layer. In this homework, we will use two FC layers with size of [64,512] and [512,512] and ReLU activation function.

The output layer is also fully-connected (with ReLU activation) with size of [512,1] to produce a single prediction value since our task is a regression problem. The error between the ground-truth value and our prediction will be used for backpropagation, which updates all trainable parameters (e.g., weight matrices of FC layers) of the MLP model. After enough iterations, we will have trained parameters and can use them for predicing ratings for new items. In our mini-batch training setting, we will compare ground-truth ratings in the training batch and our prediction values of the current batch.

Fortunately, we do not need to calculate the error and gradients for backpropagation by ourselves. PyTorch offers very simple functions for defining and training the MLP model! Please refer to the following documentation https://pytorch.org/tutorials/beginner/basics/intro.html for detailed information.

test RMSE = 0.172493
