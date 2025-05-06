#' Bagging for Linear, Logistic, Ridge, Lasso and Elastic net Models
#'
#' This function performs bagging (bootstrap aggregation) for various regression and classification models, returning final predicted values and variable importance scores where applicable.
#'
#' @param X Predictor matrix.
#' @param y Response variable.
#' @param model_type Character string indicating the type of model to use ("linear", "logistic", "ridge", "lasso", "elastic_net").
#' @param R Integer specifying the number of bootstrap samples to draw.
#' @param alpha Ridge/lasso mixing parameter (0 for ridge, 1 for lasso) for elastic net.
#' @param lambda Regularization parameter for ridge and lasso regression.
#'
#' @return A list containing the final predicted values and variable importance scores.
#' @export
#'
#' @examples
#' n <- 100
#' p <- 20
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n))
#' beta <- c(1, rep(0, p - 1))
#' y <- X %*% beta + rnorm(n)

#' predictions <- bagging(X,y,model_type = "lasso",R =100,lambda=0.2)
#' print(predictions)
#'
#' @importFrom glmnet glmnet
#' @importFrom randomForest randomForest importance
#'



bagging <- function(X, y, model_type = "linear", R = 100, alpha = NULL, lambda = NULL) {

  # Check if X and y have the same number of rows
  if (nrow(X) != length(y)) {
    warning("X and y must have the same number of rows.")
    return(NULL)
  }

  # Check if X contains NAs
  if (any(is.na(X))) {
    warning("X contains missing values. Consider imputing or removing rows with NAs.")
    return(NULL)
  }

  # Check if y contains NAs
  if (any(is.na(y))) {
    warning("y contains missing values. Consider imputing or removing rows with NAs.")
    return(NULL)
  }

  # Check if R is a valid number
  if (R <= 0) {
    warning("Please specify a valid number of bootstrap samples (R).")
    return(NULL)
  }

  # Initialize a matrix to store predicted classes from each model
  predicted_classes <- matrix(0, nrow = nrow(X), ncol = R)

  # Initialize variable importance scores
  variable_importance <- rep(0, ncol(X))

  # Loop through the bagging process R times
  for (i in 1:R) {
    # Sample with replacement
    sampled_indices <- sample(nrow(X), replace = TRUE)
    X_sampled <- X[sampled_indices, ]
    y_sampled <- y[sampled_indices]

    # Fit the model based on the chosen model type
    if (model_type == "linear") {
      model <- linear_regression(X_sampled, y_sampled, alpha = 0, lambda = 0)
    } else if (model_type == "logistic") {
      model <- logistic_regression(X_sampled, y_sampled, alpha = 1)
    } else if (model_type == "ridge") {
      model <- ridge_regression(X_sampled, y_sampled, lambda = lambda)
    } else if (model_type == "lasso") {
      model <- lasso_regression(X_sampled, y_sampled, lambda = lambda)
    } else if (model_type == "elastic_net") {
      model <- elastic_net(X_sampled, y_sampled, alpha = alpha, lambda = lambda)
    } else {
      warning("Invalid model type. Supported model types are 'linear', 'logistic', 'ridge', 'lasso', and 'elastic_net'.")
      return(NULL)
    }

    # Predict on the original data and store the predicted classes
    if (model_type == "logistic") {
      predicted_classes[, i] <- as.numeric(predict(model, newx = X, type = "class") == 1)
    } else {
      predicted_classes[, i] <- predict(model, newx = X)
    }

    # Update variable importance scores
    if (model_type != "logistic") {
      coefficients <- coef(model)[-1] # Exclude intercept
      selected_variables <- which(coefficients != 0) # Find non-zero coefficients
      variable_importance[selected_variables] <- variable_importance[selected_variables] + 1
    } else {
      coefficients <- coef(model)[-1] # Exclude intercept
      selected_variables <- which(coefficients != 0) # Find non-zero coefficients
      variable_importance[selected_variables] <- variable_importance[selected_variables] + 1
    }
  }

  # Calculate final predicted values for logistic regression using majority vote
  if (model_type == "logistic") {
    final_predicted_values <- apply(predicted_classes, 1, function(row) as.numeric(mean(row) >= 0.5))
  } else {
    final_predicted_values <- rowMeans(predicted_classes)
  }

  # Normalize variable importance scores
  variable_importance <- variable_importance / R

  return(list(predicted_values = final_predicted_values, variable_importance = variable_importance))
}
