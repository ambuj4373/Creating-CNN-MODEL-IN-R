library(reticulate)
use_python("/usr/local/bin/python3")
install.packages("tensorflow")
library(keras)
library(reticulate)
library(tensorflow)
library(imager)

# Load the CIFAR-10 dataset
cifar10 <- dataset_cifar10()
train_x <- cifar10$train$x
train_y <- cifar10$train$y
test_x <- cifar10$test$x
test_y <- cifar10$test$y

# Convert the labels to one-hot encoding
train_y <- to_categorical(train_y, num_classes = 10)
test_y <- to_categorical(test_y, num_classes = 10)

# Convert pixel values to floats and normalize to [0, 1]
train_x <- train_x / 255
test_x <- test_x / 255

# Define the CNN model architecture
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(32, 32, 3), padding = "same") %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", padding = "same") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

# Define the optimizer
optimizer <- tf$keras$optimizers$Adam(
  learning_rate = 0.001
)

# Compile the model

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer,
  metrics = c("accuracy")
)


# Train the model
history <- model %>% fit(
  train_x, train_y,
  epochs = 100,
  batch_size = 64,
  validation_data = list(test_x, test_y),
  shuffle = TRUE
)

# Evaluate the model on the test data
model %>% evaluate(test_x, test_y)

# Save the model
save_model_hdf5(model, "final_model.h5")
