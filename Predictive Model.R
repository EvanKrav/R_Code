#Clear everything from environment
rm(list = ls())

#Load all necessary libraries
library(quantmod)
library(tidyquant)
library(ggplot2)
library(TTR)
library(caret)
library(xgboost)
library(Metrics)
library(xts)
library(conflicted)
library(keras)
library(tensorflow)
library(reticulate)


# Fetch historical stock prices for NVIDIA
getSymbols("NVDA", src = "yahoo", from = "2015-01-01", to = "2024-12-31")
data <- NVDA
head(data)

# Extract adjusted closing price
stock_data <- data.frame(date = index(data), price = Cl(data))
head(stock_data)
# Check for missing values
sum(is.na(stock_data$NVDA.Close))

#Convert stock_data dataframe into XTS in order to use TTR package
stock_data_xts <- xts(stock_data$NVDA.Close, order.by = base::as.Date(stock_data$date))
colnames(stock_data_xts) <- "NVDA.Close"

# Create Moving Averages (SMA and EMA)
stock_data_xts$SMA20 <- SMA(stock_data_xts$NVDA.Close, n = 20)
stock_data_xts$EMA20 <- EMA(stock_data_xts$NVDA.Close, n = 20)

# Add Lag Features
stock_data_xts$lag_1 <- lag(stock_data_xts$NVDA.Close, 1)
stock_data_xts$lag_5 <- lag(stock_data_xts$NVDA.Close, 5)

# Calculate Returns
stock_data_xts$returns <- diff(log(stock_data_xts$NVDA.Close))

# Split the data
set.seed(123)
trainIndex <- createDataPartition(stock_data_xts$NVDA.Close, p = 0.8, list = FALSE)
trainData <- stock_data_xts[trainIndex, ]
testData <- stock_data_xts[-trainIndex, ]



# Prepare data for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -c(1)]))#, label = trainData$NVDA.Close))
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -c(1)]))#, label = testData$NVDA.Close))



# Train the model
parameters <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 6
)
model_first_version <- xgb.train(
  params = parameters,
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, test = test_matrix),
  early_stopping_rounds = 10
)

# Feature Importance
importance_matrix <- xgb.importance(model = model_first_version) #%>% xgb.plot.importance()


# Predict on Test Data
predictions <- predict(model, newdata = test_matrix)

# Evaluate Model Performance

rmse_score <- rmse(testData$price, predictions)
mae_score <- mae(testData$price, predictions)

print(paste("RMSE:", rmse_score))
print(paste("MAE:", mae_score))

# Plot Predictions vs Actual Prices

results <- data.frame(date = testData$date, actual = testData$price, predicted = predictions)
ggplot(results, aes(x = date)) +
  geom_line(aes(y = actual, color = "Actual")) +
  geom_line(aes(y = predicted, color = "Predicted")) +
  labs(title = "Stock Price Predictions", y = "Price", color = "Legend")


# Fetch New Data
getSymbols("NVDA", src = "yahoo", from = "2024-01-01", to = Sys.Date())
new_data <- data.frame(date = index(NVDA), price = Cl(NVDA))

# Update and Predict
new_data$SMA20 <- SMA(new_data$price, n = 20)
new_data$EMA20 <- EMA(new_data$price, n = 20)
new_data$lag_1 <- lag(new_data$price, 1)
new_data$lag_5 <- lag(new_data$price, 5)
new_data <- na.omit(new_data)

new_matrix <- xgb.DMatrix(data = as.matrix(new_data[, -c(1, 2)]))
new_predictions <- predict(model, newdata = new_matrix)

# Add to Results
new_data$predicted <- new_predictions


# Compare new predictions to actual prices
accuracy <- data.frame(
  actual = new_data$price,
  predicted = new_data$predicted,
  error = new_data$price - new_data$predicted
)

# Calculate Dynamic RMSE
dynamic_rmse <- sqrt(mean(accuracy$error^2))
print(paste("Dynamic RMSE:", dynamic_rmse))


##################################################################################

#LSTM Model use BTC price for the sake of forecasting

getSymbols("BTC-USD", src = "yahoo", from = "2015-01-01", to = "2024-12-31")

# Extract closing prices
bitcoin_prices <- `BTC-USD`[, "BTC-USD.Close"]

# Plot the Bitcoin closing prices
plot(bitcoin_prices, main = "Bitcoin Closing Prices", col = "blue")

# Step 2: Preprocess Data
# Convert to a data frame for easier manipulation
bitcoin_df <- data.frame(date = index(bitcoin_prices), price = coredata(bitcoin_prices))

# Remove NA values
bitcoin_df <- na.omit(bitcoin_df)

# Normalize the price data (scaling between 0 and 1)
max_price <- max(bitcoin_df$BTC.USD.Close)
min_price <- min(bitcoin_df$BTC.USD.Close)
bitcoin_df$scaled_price <- (bitcoin_df$BTC.USD.Close - min_price) / (max_price - min_price)

# Create a function to prepare time series data for LSTM
create_sequences <- function(data, time_steps = 30) {
  x <- list()
  y <- list()
  for (i in seq_len(nrow(data) - time_steps)) {
    x[[i]] <- as.matrix(data[i:(i + time_steps - 1), "scaled_price"])
    y[[i]] <- data[i + time_steps, "scaled_price"]
  }
  list(
    x = array(unlist(x), dim = c(length(x), time_steps, 1)),
    y = unlist(y)
  )
}

# Define the number of time steps (e.g., 30 days)
time_steps <- 30

# Prepare the training and testing datasets
split_index <- floor(0.8 * nrow(bitcoin_df))
train_data <- bitcoin_df[1:split_index, ]
test_data <- bitcoin_df[(split_index + 1):nrow(bitcoin_df), ]

train_seq <- create_sequences(train_data, time_steps)
test_seq <- create_sequences(test_data, time_steps)
use_virtualenv("C:/Users/evang/Documents/.virtualenvs/r-tensorflow", required = TRUE)

# Step 3: Build the LSTM Model
# Define time_steps
time_steps <- 30  # Example time steps, adjust as per your data

# Build the LSTM Model
model <- keras_model_sequential() %>%
layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(time_steps, 1)) %>%
layer_dropout(rate = 0.2) %>%
layer_lstm(units = 50, return_sequences = FALSE) %>%
layer_dropout(rate = 0.2) %>%
layer_dense(units = 1)


# Compile the model
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("mean_absolute_error")
)

# Step 4: Train the Model
history <- model %>% fit(
  x = train_seq$x,
  y = train_seq$y,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2
)

# Plot training history
plot(history)

# Step 5: Evaluate the Model
model %>% evaluate(test_seq$x, test_seq$y)

# Step 6: Make Predictions
predictions <- model %>% predict(test_seq$x)

# Rescale predictions back to original price range
predicted_prices <- predictions * (max_price - min_price) + min_price
actual_prices <- test_seq$y * (max_price - min_price) + min_price

# Step 7: Visualize Results
results <- data.frame(
  date = bitcoin_df$date[(split_index + time_steps + 1):nrow(bitcoin_df)],
  actual = actual_prices,
  predicted = predicted_prices
)

# Plot actual vs predicted prices
ggplot(results, aes(x = date)) +
  geom_line(aes(y = actual, color = "Actual")) +
  geom_line(aes(y = predicted, color = "Predicted")) +
  labs(title = "Bitcoin Price Prediction with LSTM", x = "Date", y = "Price") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()


