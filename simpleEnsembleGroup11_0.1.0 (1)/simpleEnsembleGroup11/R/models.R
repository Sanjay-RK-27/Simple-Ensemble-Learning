#' Linear Regression
#'
#' The linear_regression function implements linear regression using the glmnet package. Linear regression is a widely used statistical model for predicting a continuous response variable based on one or more predictor variables.
#'
#' @param X Predictor matrix.
#' @param y Response variable.
#' @param bagging Logical indicating whether to perform bagging.
#' @param prescreen Logical indicating whether to pre-screen predictors.
#' @param alpha Ridge/lasso mixing parameter (0 for ridge, 1 for lasso).
#' @param lambda Regularization parameter.
#' @param ... Additional arguments passed to other functions.
#'
#' @return The fitted linear regression model.
#' @export
#'
#' @examples
#' # Generate sample data for a linear regression problem
#' n <- 100 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' y <- X %*% beta + rnorm(n) # Generate the response variable
#'
#' ## Linear regression without bagging or prescreening
#' # Fit the Linear regression model without bagging or prescreening
#' model1 <- linear_regression(X, y, bagging = FALSE, prescreen = FALSE, alpha = 0, lambda = 0.1)
#'
#' # Make predictions on the original data
#' predictions1 <- predict(model1, newx = X)
#'
#' print(head(predictions1)
#'
#'
#'
#' @importFrom glmnet glmnet
#' @importFrom randomForest randomForest
linear_regression <- function(X, y, bagging = FALSE, prescreen = FALSE, alpha = 0, lambda = 0.1) {

  # Check if X is a matrix or data frame
  if (!is.matrix(X) && !is.data.frame(X)) {
    warning("X must be a matrix or data frame.")
    return(NULL)
  }

  if (!is.numeric(X) || !is.numeric(y)) {
    warning("X and y must be numeric.")
    return(NULL)
  }

  # Check if X has at least one column
  if (ncol(X) == 0) {
    warning("X must have at least one column.")
    return(NULL)
  }

  # Check if y has at least one row
  if (length(y) == 0) {
    warning("y must have at least one row.")
    return(NULL)
  }


  # If X and y have different number of rows
  if (nrow(X) != length(y)) {
    warning("X and y must have the same number of rows.")
    return(NULL)
  }

  # If X contains NAs
  if (anyNA(X)) {
    stop("X contains missing values. Consider imputing or removing rows with NAs.")
  }

  # If y contains NAs
  if (anyNA(y)) {
    stop("y contains missing values. Consider imputing or removing rows with NAs.")
  }

  if (prescreen) {
    # Check if prescreen is a valid number
    if (missing(prescreen) || prescreen <= 0 || prescreen >= ncol(X)) {
      warning("Please specify a valid number of predictors to keep when using prescreen.")
      return(NULL)
    }
    x <- prescreen_predictors(X, y, prescreen)
  } else {
    x <- X
  }

  if (bagging) {
    # Check if bagging is a valid number
    if (bagging <= 0) {
      warning("Please specify a valid number of bootstrap samples for bagging.")
      return(NULL)
    }
    model <- bagging(x, y, model_type = "linear", R= bagging, alpha = alpha, lambda = lambda)
  } else {
    if (alpha < 0 || alpha > 1) {
      warning("alpha must be between 0 and 1.")
      return(NULL)
    }

    if (lambda <= 0) {
      warning("lambda must be positive.")
    }

    if (alpha != 0 || lambda != 0.1) {
      warning("alpha and lambda are only used when bagging is set to TRUE.")
    }

    model <- glmnet(x, y, alpha = alpha, lambda = lambda)
  }

  return(model)
}

#' Logistic Regression
#'
#' The logistic_regression function implements logistic regression using the glmnet package. Logistic regression is a widely used statistical model for binary classification problems, where the goal is to predict a binary outcome (e.g., 0 or 1, success or failure) based on a set of predictor variables.
#'
#' @inheritParams linear_regression
#'
#' @return The fitted logistic regression model.
#' @export
#'
#' @examples
#' #'# Generate sample data for a binary classification problem
#' n <- 100 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' prob <- 1 / (1 + exp(-X %*% beta)) # Calculate the probability of the binary outcome
#' y <- factor(rbinom(n * 1, 1, prob)) # Generate the binary outcome variable

#' ## Logistic regression without bagging or prescreening
#' # Fit the logistic regression model without bagging or prescreening
#' model1 <- logistic_regression(X, y, bagging = FALSE, prescreen = FALSE, alpha = 0, lambda = 0.1)

#' # Make predictions on the original data
#' predictions1 <- predict(model1, newx = X)
#'
#' print(head(predictions1))
#'
#'
#' @importFrom glmnet glmnet
logistic_regression <- function(X, y, bagging = FALSE, family = "binomial", prescreen = FALSE, alpha = 0.5, lambda = 0.1) {

  # Check if the glmnet package is installed, and install it if not
  if (!require(glmnet, quietly = TRUE)) {
    warning("The glmnet package is required for this function. Installing it now.")
    install.packages("glmnet")
    library(glmnet)
  }

  # Check if X and y have the same number of rows
  if (nrow(X) != length(y)) {
    warning("X and y must have the same number of rows.")
    return(NULL)
  }

  # Check if X and y have the same data types
  if (!is.numeric(X) || !is.factor(y)) {
    warning("X must be numeric and y must be a factor.")
    return(NULL)
  }

  # Check if X has at least one column
  if (ncol(X) == 2) {
    warning("X must have at least two column.")
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


  if (prescreen) {
    # Check if prescreen is a valid number
    if (missing(prescreen) || prescreen <= 0 || prescreen >= ncol(X)) {
      warning("Please specify a valid number of predictors to keep when using prescreen.")
      return(NULL)
    }
    X <- prescreen_predictors(X, y, prescreen)
  }

  if (bagging) {
    # Check if bagging is a valid number
    if (bagging <= 0 || bagging >= nrow(X)) {
      warning("Please specify a valid number of bootstrap samples for bagging.")
      return(NULL)
    }
    model <- bagging(X, y, model_type = "logistic",R= bagging, alpha = alpha, lambda = lambda)
  } else {
    if (alpha < 0 || alpha > 1) {
      warning("alpha must be between 0 and 1.")
      return(NULL)
    }

    if (lambda <= 0) {
      warning("lambda must be positive.")
      return(NULL)
    }

    if (family != "binomial") {
      warning("Only the 'binomial' family is supported for logistic regression.")
      return(NULL)
    }

    model <- glmnet(X, y, family = family, alpha = alpha, lambda = lambda)
  }

  return(model)
}

#' Ridge Regression
#'
#' The ridge_regression function implements ridge regression using the glmnet package. Ridge regression is a type of regularized linear regression that adds a penalty term to the least squares objective function, which helps to reduce the variance of the model and prevent overfitting.
#'
#' @inheritParams linear_regression
#'
#' @return The fitted ridge regression model.
#' @export
#'
#' @examples
#'
#' #' # Generate sample data for a Ridge regression problem
#' n <- 100 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' y <- X %*% beta + rnorm(n) # Generate the response variable

#' ## Linear regression without bagging or prescreening
#' # Fit the Linear regression model without bagging or prescreening
#' model1 <- ridge_regression(X, y, bagging = FALSE, prescreen = FALSE, lambda = 0.1)

#' # Make predictions on the original data
#' predictions1 <- predict(model1, newx = X)
#' print(head(predictions1))
#'
#' @importFrom glmnet glmnet

ridge_regression <- function(X, y, bagging = FALSE, prescreen = FALSE, lambda = 0.1) {

  # Check if the glmnet package is installed, and install it if not
  if (!require(glmnet, quietly = TRUE)) {
    warning("The glmnet package is required for this function. Installing it now.")
    install.packages("glmnet")
    library(glmnet)
  }

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

  if (prescreen) {
    # Check if prescreen is a valid number
    if (missing(prescreen) || prescreen <= 0 || prescreen >= ncol(X)) {
      warning("Please specify a valid number of predictors to keep when using prescreen.")
      return(NULL)
    }
    X <- prescreen_predictors(X, y, prescreen)
  }

  if (bagging) {
    # Check if bagging is a valid number
    if (bagging <= 0) {
      warning("Please specify a valid number of bootstrap samples for bagging.")
      return(NULL)
    }

    # Check if the response variable is binary
    if (length(unique(y)) == 2) {
      model <- bagging(X, y, model_type = "logistic", R= bagging, alpha = 0, lambda = lambda)
    } else {
      model <- bagging(X, y, model_type = "linear", R= bagging, alpha = 0, lambda = lambda)
    }
  } else {
    if (lambda <= 0) {
      warning("lambda must be positive.")
      return(NULL)
    }

    # Check if the response variable is binary
    if (length(unique(y)) == 2) {
      model <- glmnet(X, y, family = "binomial", alpha = 0, lambda = lambda)
    } else {
      model <- glmnet(X, y, alpha = 0, lambda = lambda)
    }
  }

  return(model)
}

#' Lasso Regression
#'
#' The lasso_regression function implements lasso regression using the glmnet package. Lasso regression is a type of regularized linear regression that adds a penalty term to the least squares objective function, which helps to perform feature selection by shrinking some coefficients to exactly zero.
#'
#' @inheritParams ridge_regression
#'
#' @return The fitted lasso regression model.
#' @export
#'
#' @examples
#' # Generate sample data for a Lasso regression problem
#' n <- 100 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' y <- X %*% beta + rnorm(n) # Generate the response variable

#' ## Linear regression without bagging or prescreening
#' # Fit the Linear regression model without bagging or prescreening
#' model1 <- lasso_regression(X, y, bagging = FALSE, prescreen = FALSE, lambda = 0.1)

#' # Make predictions on the original data
#' predictions1 <- predict(model1, newx = X)

#' print(head(predictions1))
#'
#' @importFrom glmnet glmnet
lasso_regression <- function(X, y, bagging = FALSE, prescreen = FALSE, lambda = 0.1) {

  # Check if the glmnet package is installed, and install it if not
  if (!require(glmnet, quietly = TRUE)) {
    warning("The glmnet package is required for this function. Installing it now.")
    install.packages("glmnet")
    library(glmnet)
  }

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


  if (prescreen) {
    # Check if prescreen is a valid number
    if (missing(prescreen) || prescreen <= 0 || prescreen >= ncol(X)) {
      warning("Please specify a valid number of predictors to keep when using prescreen.")
      return(NULL)
    }
    X <- prescreen_predictors(X, y, prescreen)
  }

  if (bagging) {
    # Check if bagging is a valid number
    if (bagging <= 0 ) {
      warning("Please specify a valid number of bootstrap samples for bagging.")
      return(NULL)
    }

    # Check if the response variable is binary
    if (length(unique(y)) == 2) {
      model <- bagging(X, y, model_type = "logistic", R= bagging, alpha = 1, lambda = lambda)
    } else {
      model <- bagging(X, y, model_type = "linear", R= bagging, alpha = 1, lambda = lambda)
    }
  } else {
    if (lambda <= 0) {
      warning("lambda must be positive.")
      return(NULL)
    }

    # Check if the response variable is binary
    if (length(unique(y)) == 2) {
      model <- glmnet(X, y, family = "binomial", alpha = 1, lambda = lambda)
    } else {
      model <- glmnet(X, y, alpha = 1, lambda = lambda)
    }
  }

  return(model)
}

#' Elastic Net
#'
#' The elastic_net function implements elastic net regression using the glmnet package. Elastic net is a regularized linear regression technique that combines the properties of both ridge regression and lasso regression. It can perform feature selection and coefficient shrinkage simultaneously.
#'
#' @inheritParams linear_regression
#'
#' @return The fitted elastic net model.
#' @export
#'
#' @examples
#' #' # Generate sample data for a elastic net  problem
#' n <- 100 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' y <- X %*% beta + rnorm(n) # Generate the response variable
#'
#' ## Linear regression without bagging or prescreening
#' # Fit the Linear regression model without bagging or prescreening
#' model1 <- elastic_net(X, y, bagging = FALSE, prescreen = FALSE, lambda = 0.1)
#'
#' # Make predictions on the original data
#' predictions1 <- predict(model1, newx = X)
#'
#' print(head(predictions1))
#'
#' @importFrom glmnet glmnet

elastic_net <- function(X, y, bagging = FALSE, prescreen = FALSE, alpha = 0.5, lambda = 0.1) {

  # Check if the glmnet package is installed, and install it if not
  if (!require(glmnet, quietly = TRUE)) {
    warning("The glmnet package is required for this function. Installing it now.")
    install.packages("glmnet")
    library(glmnet)
  }

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

  if (prescreen) {
    # Check if prescreen is a valid number
    if (missing(prescreen) || prescreen <= 0 || prescreen >= ncol(X)) {
      warning("Please specify a valid number of predictors to keep when using prescreen.")
      return(NULL)
    }
    X <- prescreen_predictors(X, y, prescreen)
  }

  if (bagging) {
    # Check if bagging is a valid number
    if (bagging <= 0) {
      warning("Please specify a valid number of bootstrap samples for bagging.")
      return(NULL)
    }

    # Check if the response variable is binary
    if (length(unique(y)) == 2) {
      model <- bagging(X, y, model_type = "logistic", R= bagging, alpha = alpha, lambda = lambda)
    } else {
      model <- bagging(X, y, model_type = "linear", R= bagging, alpha = alpha, lambda = lambda)
    }
  } else {
    if (alpha < 0 || alpha > 1) {
      warning("alpha must be between 0 and 1.")
      return(NULL)
    }

    if (lambda <= 0) {
      warning("lambda must be positive.")
      return(NULL)
    }

    # Check if the response variable is binary
    if (length(unique(y)) == 2) {
      model <- glmnet(X, y, family = "binomial", alpha = alpha, lambda = lambda)
    } else {
      model <- glmnet(X, y, alpha = alpha, lambda = lambda)
    }
  }

  return(model)
}


#' Random Forest
#'
#' The random_forest function implements random forest for regression or classification using the randomForest package. Random forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and stability of the predictions.
#'
#' @param X Predictor matrix.
#' @param y Response variable.
#' @param prescreen Logical indicating whether to pre-screen predictors.
#' @param is_binary Logical indicating whether the response variable is binary (classification) or continuous (regression).
#' @param ntree Number of trees to grow (default is 500).
#' @param mtry Number of variables randomly sampled as candidates at each split (default is square root of number of predictors for classification, and one-third for regression).
#'
#'
#' @return The fitted random forest model.
#' @export
#'
#' @examples
#' #' # Generate sample data for a random Forest problem
#' n <- 100 # Number of observations
#' p <- 10 # Number of predictors
#' X <- as.matrix(matrix(rnorm(n * p), nrow = n)) # Generate predictor matrix
#' beta <- c(1, rep(0, p - 1)) # True coefficients (first one is 1, others are 0)
#' y <- X %*% beta + rnorm(n) # Generate the response variable

#' ## Random Forest without bagging or prescreening
#' # Fit the Random Forest regression model without bagging or prescreening
#' model1 <- random_forest(X, as.vector(y), prescreen = FALSE,is_binary = FALSE )

#' # Make predictions on the original data
#' predictions1 <- predict(model1, newx = X)

#' print(head(predictions1))
#'
#'
#' @importFrom randomForest randomForest importance
random_forest <- function(X, y, prescreen = FALSE, is_binary = TRUE, ntree = 500, mtry = NULL) {

  # Check if the randomForest package is installed, and install it if not
  if (!require(randomForest, quietly = TRUE)) {
    warning("The randomForest package is required for this function. Installing it now.")
    install.packages("randomForest")
    library(randomForest)
  }

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


  if (prescreen) {
    # Check if prescreen is a valid number
    if (missing(prescreen) || prescreen <= 0 || prescreen >= ncol(X)) {
      warning("Please specify a valid number of predictors to keep when using prescreen.")
      return(NULL)
    }
    X <- prescreen_predictors(X, y, prescreen)
  }

  if (is.null(mtry)) {
    if (is_binary) {
      mtry <- floor(sqrt(ncol(X)))
    } else {
      mtry <- max(floor(ncol(X) / 3), 1)
    }
  } else {
    # Check if mtry is a valid number
    if (mtry <= 0 || mtry >= ncol(X)) {
      warning("Please specify a valid number of variables to sample at each split (mtry).")
      return(NULL)
    }
  }

  # Check if ntree is a valid number
  if (ntree <= 0) {
    warning("Please specify a valid number of trees (ntree).")
    return(NULL)
  }

  if (is_binary) {
    model <- randomForest(X, as.factor(y), ntree = ntree, mtry = mtry)
  } else {
    model <- randomForest(X, y, ntree = ntree, mtry = mtry)
  }

  return(model)
}
