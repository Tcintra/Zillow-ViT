# Implementing a Visual Transformer (ViT) on Zillow Housing Data

_This is an educational project to investigate how we can improve NNs to make them better at learning from images._

Let's try implementing a transformer model on the Zillow housing data. We will take an iterative approach, starting with a simple\[r\] CNN and enhancing it into a transformer model. Over the course of this notebook, we will include new optimizations that will gradually improve the performance of the model, such as using adaptive stochastic gradient descent algorithms (Adam), implementing Dropout layers (to reduce overfitting), and parallelizing computation to speed up our training. Once we have benchmarked our transformer model trained exclusively on Zillow data, we will then fine-tune a pre-trained model from Hugging Face using the same data and compare the performance of the two models.

Our model's task is to classify images from a house into 3 buckets:

- Cheap (bottom 33\% of observed prices).
- Average (middle 33\% of observed prices).
- Expensive (top 33\% of observed prices).

## A Simple CNN

Let's begin with a simple classification model (CNN, no attention).

### Preprocessing

We first preprocess our data into something a CNN could actually work with. We have a large JSON file containing the ID, price, and a set of URLs for each house (3044 houses). We then fetch the images contained in each of these URLs and preprocess (vectorize) them into tensors, and then we resize and normalize them. We will resize our images to be a standard (224, 224, 3) tensor, meaning a 224 x 224 pixes where each pixel is an RGB vector.

Since we are using a simple CNN, it's difficult to handle multiple images for each house, since we need feature vectors of fixed size. We could concatenate the images into large tensors, and pad the tensors for houses with less images so that they are all the same size. However, we will stick to a single image per house at first, and then venture into multiple images per house with a Transformer model, which can handle feature vectors of varying sizes.

We format our targets/labels using a zero-indexed integer encoding (cheap = 0, average = 1, expensive = 2). 

We will use Pytorch's `DataLoader` to load our training data in batches. This allows us to perform mini-batched gradient descent, while also preventing us from having to load the entire dataset into memory or having to train on a single data point at a time.

### Model Architecture

We will begin with a CNN that contains two convolutional layers followed by 2 dense layers. The convolutional layers will learn spatial features about the images, such as texture, bright spots, edges, etc., and output these feature maps. This means the output of a convolutional layer is "3 dimensional" since each feature vector says something about a "chunk" or "window" of the image. We then apply a Rectified Linear Unit (ReLU) activation function to our feature map, effectively truncating all negative values for each of our features but retaining the output shape.

The first dense layer then flattens the 3-dimensional feature map from the last convolutional layer into a 2-dimensional tensor, connecting the features from all chunks of the image (hense, it's "dense"). The second dense layer then converts this flattened tensor into a scalar output for each of our categories (cheap, average, expensive) using the softmax activation function. The softmax helps us discern which of the categories are more likely (higher value).

Notice that for each CNN we may also choose to reduce the dimensionality of the output. We do this to reduce training times and prevent overfitting. The actual output of a convolutional layer splits the image into $k \times k$ chunks, where $k$ is the "kernel size" (or window size) we use to analyze our image. We can optionally choose to down-sample our data by taking the maximum of each feature over some set of chunks. For example, we can have a bunch of chunks of size $3 \times 3$, and we may downsample our data by taking maximum of each feature for every $2 \times 2$ chunks, meaning we cut our data in half. In this example, we have 4 chunks, each of size $3 \times 3$, being collapsed down into a single $3 \times 3$ chunk. This is known as "Max Pooling".

| Layer | Input Dimensions | Output Dimensions | Objective | Activation Function | In Channels | Out Channels | Kernel | Stride | Padding |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------| 
| Convolutional Layer 1 | 224, 224, 3 | 224, 224, 32 | Learn spatial features | ReLU | 3 | 32 | 3 | 1 | 1
| Pooling Layer 1 | 224, 224, 32 | 112, 112, 32 | Down-sampling | | 32 | 32 | 2 | 2 | 0
| Convolutional Layer 2 | 112, 112, 32 | 112, 112, 64 | Learn spatial features | ReLU | 32 | 64 | 3 | 1 | 1
| Pooling Layer 2 | 112, 112, 64 | 56, 56, 64 | Down-sampling | | 64 | 64 | 2 | 2 | 0
| Dense Layer 1 | 200704 | 512 | Learn spatial features | ReLU | 200704 | 512 | | | |
| Dense Layer 2 | 512 | 3 | Classification | Softmax | 512 | 3 | | | |

As an example, we begin with $h \times w \times 3$ tensor, where $h$ is the height of the image and $w$ is the width, usually $224 \times 224$, and $3$ merely denotes the input features we know about each point in our tensor (RGB). Our model then does the following:

1. **Convolutional Layer 1:** scans each possible chunk of shape (3, 3) in the image tensor and produce a feature vector of length 32 for each chunk (think of a sliding window of height=width=kernel size). This learns spatial features about each chunk. If we raise the `stride` parameter we end up "skipping" pixel, effectively down-sampling the data. We pad the image with zeros at the border to ensure we don't drop the edge pixels/chunks.
2. **Pooling Layer 1:** We down-sample the output feature map from the previous layer by taking the max of each feature for every (2, 2) block of (3, 3) chunks.
3. **Convolutional Layer 2:** we effectively repeat the spatial learning, this time with more features (adding some complexity).
4. **Pooling Layer 2:** We down-sample again. Notice that if we didn't downsample, we would end up with a 3.2M input vector to the first dense layer, producing 1.6Bn parameters, as opposed to the 100M we actually use (a 96\% reduction).
5. **Dense Layer 1:** We flatten the feature map and connect all of the nodes, meaning we can now infer something about the image as a whole, using 512 nodes (each having about 200k weights). This is basically a dot product between the flattened feature map from Pooling Layer 2 and the weights (plus bias) in each of the 512 nodes.
6. **Dense Layer 2:** Finally, we take the output of the first dense layer and try to classify the image into our cheap, average, and expensive categories. Each category is represented by a node with 512 weight parameters. The node essentially guesses whether the 512 features make it likely to belong to that node, or unlikely to belong to that node. We then apply the `softmax` activation function to make our outputs compatible with a negative log likelhood loss function (similar to a sigmoid activation for a single-class classifier). 

### Scoring

During training, Pytorch's `CrossEntropyLoss` will apply the softmax activation function to our output automatically, and then compute a log-likelihood loss against the label. Under the hood, `CrossEntropyLoss` is really just a negative log-likelihood implementation that is generalized to a multi-class classification problem, where the softmax function is used instead of the usual sigmoid before calculating the log likelihood loss. During inference, we instead will directly take the `argmax` to classify the image, since softmax is monotonic and would not alter our prediction.

Our goal in training is to minimize the `CrossEntropyLoss` by updating the parameters in each of our convolutional and dense layers. This effectively aims to maximize the chances that the output probabilities from the softmax activation function match the actual label in our dataset. 

### Training

We perform training using gradient descent. We will run our optimization using the `Adam` optimizer right off the bat, since it's faster, better, and easier to use than implementing our own gradient descent. The `Adam` optimizer will compute the `CrossEntropyLoss` at each iteration, determine the gradient of our loss function, and update the model's parameters such that we "walk down" the gradient towards a global minimum. The main differences between `Adam` and a simple gradient descent approach is that it: 

- Trains the model on "mini-batches" of data at each step instead of the entire training set, allowing the model to get better without having to see the entire dataset all at once. This is also known as Stochastic Gradient Descent (SGD).
- Applies an exponential smoothing (EMA) to the gradient, meaning the gradient we use to update our weights at each step account for the gradients we computed at previous steps, giving us some "momentum" in our descent.
- Calculates an EMA of the "magnitude" of our gradient and uses it to dampen our learning rate at each step. This effectively prevents us from moving too quickly when the gradient is "steep" or moving too slowly when the gradient is "flat".

In a nutshell, we do the following at each step:

1. **Forward Propagation:** compute probabilities of the image belonging to each of our classes (the output of the final layer).
2. **Compute Loss:** compute the `CrossEntropyLoss` of our prediction from step (1) against the true label.
3. **Backward Propagation:** using the `Adam` algorithm, update the weights of our network using the gradient of the loss from step (2).

We will then run these steps for some number of iterations.

## A Vision Transformer [To Do]

Once we have trained and validated our simple CNN, we will implement the "encoder" attention blocks that turn our simple CNN into a transformer model. By doing so, we can now feed multiple images as a feature set for each house. This is because transformer models are able to handle feature vectors of varying sizes, using the "attention" blocks to discern which images contain the most information about prices. We will then compare the validation score of our transformer model to our simple CNN.

## Piggybacking off Pre-Trained Models [To Do]

Finally, we will pull a pre-trained transformer model from HuggingFace and deploy it on our dataset. In this case, instead of training our model exclusively on the Zillow data, we leverage the idea of **transfer learning** to use pre-trained features from millions of other stock images. The pre-trained model has already learned how to extract useful features from many images, and we can use its architecture to extract these same features about our house images. We then extend (fine-tune) our model using these new pre-trained features. In this process, we unfreeze the weights from the pre-trained model and adapt them (via gradient descent) to our new classification image. In this process, we introduce a few additional layers to our model to adapt it to our image classification problem. In particular, we add a dense layer at the end of our NN that transforms the output features into a scalar value, which we then use for classification (e.g. 0.1 -> cheap, 0.9 -> expensive).
