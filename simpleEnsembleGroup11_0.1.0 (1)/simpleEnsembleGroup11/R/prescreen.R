#' Prescreen Top K Predictors
#'
#' This function prescreens the top K most informative predictors based on
#' the variable importance scores from a Random Forest model.
#'
#' @param X A matrix where each column represents a predictor variable.
#' @param y A matrix with a single column representing the response variable.
#' @param K An integer specifying the number of top predictors to select (default is 7).
#'
#' @return A matrix containing the top K predictors selected based on the Random Forest
#'   variable importance scores.
#'
#' @importFrom randomForest randomForest importance
#'
#' @examples
#' # Generate some example data
#' set.seed(123)
#' n <- 100
#' p <- 2000
#' X <- matrix(rnorm(n * p), nrow = n)
#' y <- matrix(rnorm(n), ncol = 1)
#' # Prescreen top 5 predictors
#' top_predictors <- prescreen_predictors(X, as.vector(y), K = 20)
#'
#' print(head(top_predictors))
#' #For binary response
#' y_bin <- matrix(sample(c(0, 1), n, replace = TRUE), ncol = 1)
#' top_predictors_bin <- prescreen_predictors(X,as.factor(y_bin), K = 20)
#'
#' print(head(top_predictors_bin))
#'

prescreen_predictors <- function(X, y, K = 7) {

  # Check if X and y have the same number of rows
  if (nrow(X) != length(y)) {
    warning("X and y must have the same number of rows.")
    return(NULL)
  }

  # Determine the number of predictors
  p <- ncol(X)

  # Check if the number of predictors is less than or equal to K
  if (p <= K) {
    warning("The number of predictors 'p' is less than or equal to the number 'K' of top predictors requested. Returning all predictors.")
    return(X)
  }

  # Check if K is a valid number
  if (missing(K) || K <= 0 || K >= p) {
    warning("Please specify a valid number of predictors to keep (K).")
    return(NULL)
  }

  # Ensure the randomForest package is available
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("The 'randomForest' package is required but not available.")
  }

  # Check if X contains NAs
  if (any(is.na(X))) {
    warning("X contains missing values. Consider imputing or removing rows with NAs.")
  }

  # Check if y contains NAs
  if (any(is.na(y))) {
    warning("y contains missing values. Consider imputing or removing rows with NAs.")
  }

  # Fit a Random Forest model to compute importance
  rf_model <- randomForest::randomForest(X, y)

  # Retrieve the variable importance scores
  importance_scores <- importance(rf_model)

  # Select the top K predictors based on importance
  top_predictors_idx <- order(importance_scores, decreasing = TRUE)[1:K]
  top_predictors <- X[, top_predictors_idx, drop = FALSE]

  return(top_predictors)
}
