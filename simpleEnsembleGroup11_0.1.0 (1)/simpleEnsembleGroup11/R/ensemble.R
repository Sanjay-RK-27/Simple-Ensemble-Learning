#' Ensemble Learning
#'
#' This function performs ensemble learning by combining the predictions of multiple models. It supports both regression and classification tasks by appropriately weighting and aggregating model predictions.
#'
#' @param X Predictor matrix.
#' @param y Response variable.
#' @param models List of model functions to be combined in the ensemble. The functions should take X and y as input and return a fitted model object.
#' @param weights Numeric vector of weights for each model in the ensemble. The weights should sum to 1.
#'
#' @return The ensemble model and the final predictions.
#' @export
#'
#' @examples
#' # Generate sample data for a binary classification problem
#' n <- 10000 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' prob <- 1 / (1 + exp(-X %*% beta)) # Calculate the probability of the binary outcome
#' y <- factor(rbinom(n * 1, 1, prob)) # Generate the binary outcome variable
#'
#' ensemble_model <- ensemble_learning(X, y,
#'                                     models = list(elastic_net,random_forest),
#'                                     weights = c(0.3, 0.3, 0.4))
#'
#' new_predictions <- ensemble_model$final_predictions
#'
#' print(head(new_predictions,100))

ensemble_learning <- function(X, y, models, weights) {
  if (sum(weights) != 1) {
    stop("Weights must sum to 1.")
  }

  n <- nrow(X)
  ensemble_predictions <- matrix(0, nrow = n, ncol = length(models))

  for (i in seq_along(models)) {
    model <- models[[i]](X, y)
    ensemble_predictions[, i] <- predict(model, X)
  }

  # Determine if the response variable is binary or continuous
  is_binary <- length(unique(y)) == 2

  if (is_binary) {
    # Binary response variable
    final_predictions <- apply(ensemble_predictions, 1, function(x) {
      which.max(x) - 1  # Convert probabilities to binary predictions
    })
  } else {
    # Continuous response variable
    final_predictions <- rowSums(ensemble_predictions * weights)
  }

  return(list(final_predictions = final_predictions, ensemble_model = ensemble_predictions))
}
