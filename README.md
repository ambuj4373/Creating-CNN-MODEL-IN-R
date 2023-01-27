# Creating-CNN-MODEL-IN-R


# Load required packages
library(tensorflow)
library(keras)
library(jpeg)
# Set working directory
setwd(choose.dir())
# Get a list of the file paths for all of the images in the "smile" and "non_smile" directories
smile_image_files <- list.files("smile", full.names = TRUE)
non_smile_image_files <- list.files("non_smile", full.names = TRUE)
# Read in the images from the "smile" and "non_smile" directories
smiling_images <- lapply(smile_image_files, readJPEG)
non_smiling_images <- lapply(non_smile_image_files, readJPEG)
# Combine the smiling and non-smiling images into a single dataset
images <- c(smiling_images, non_smiling_images)

# Load the caret package
library(caret)
set.seed(123)
train_index <- createDataPartition(1:length(images), p = 0.8, list = FALSE)
train_set <- images[train_index]
test_set <- images[-train_index]
# Split the dataset into a training set and a testing set
train_set <- lapply(train_set, function(x) as.array(x))
test_set <- lapply(test_set, function(x) as.array(x))
# Normalize the pixel values
train_set <- lapply(train_set, function(x) x / 255)
test_set <- lapply(test_set, function(x) x / 255)
# Create the labels for the training and testing sets
train_labels <- c(rep(0, length(non_smiling_images)), rep(1, length(smiling_images)))
test_labels <- ifelse(length(test_set) > length(non_smiling_images),
                      c(rep(0, length(non_smiling_images)), rep(1, length(test_set) - length(non_smiling_images))),
                      rep(0, length(test_set)))
#Createting train_set_tensor to convert into compatible dimesnion
train_set_tensor <- array_reshape(train_set, c(length(train_set), 64, 64, 3))
test_set_tensor <- array_reshape(test_set, c(length(test_set), 64, 64, 3))
# Convert the images in train_set to magick image objects
train_labels_cat <- to_categorical(train_labels)
test_labels_cat <- to_categorical(test_labels)
# Build the CNN model

library(magrittr)
library(tensorflow)
library(keras)
# Define the model architecture

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(64, 64, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)
# Fit the model using the train_set_tensor tensor and the train_labels vector
history <- model %>% fit(
  x = train_set_tensor,
  y = train_labels,
  epochs = 32,
  batch_size = 3,
  validation_split = 0.2
)
#EARLY STOPPING TO AVOID OVERFITTING
early_stopping <- keras::callback_early_stopping(monitor = "val_loss", min_delta = 0.01, patience = 5)
history <- model %>% fit(
  x = train_set_tensor,
  y = train_labels,
  epochs = 10,
  batch_size = 1,
  validation_split = 0.2,
  callbacks = list(early_stopping)
)
history_df <- as.data.frame(history)
history_df$epoch <- 1:nrow(history_df)
library(ggplot2)
ggplot(history_df, aes(x = epoch, y = value)) +
  geom_line(aes(color = metric)) +
  facet_wrap(~ metric, scales = "free_y")
predictions <- model %>% predict(test_set)
predictions <- model %>% predict(test_set_tensor)
Summary(predictions)
#--------------------------------------------------
