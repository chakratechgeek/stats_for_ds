# Linear Regression Methods
There are several other methods and variations in linear regression and general regression analysis. Each method has its strengths and is best suited for specific situations, depending on the nature of the data and the problem you're trying to solve. Here are some commonly used regression methods:

### 1. **Ordinary Least Squares (OLS)**
   - **Use Case:** When the relationship between the independent variables and the dependent variable is linear, and the errors (residuals) are normally distributed with constant variance (homoscedasticity). 
   - **Why Use It:** OLS provides the best linear unbiased estimators (BLUE) under these conditions.

### 2. **Weighted Least Squares (WLS)**
   - **Use Case:** When the variance of the errors is not constant (heteroscedasticity), meaning that some observations are more variable than others.
   - **Why Use It:** WLS assigns weights to each observation to account for different variances, leading to more accurate estimates.

### 3. **Generalized Least Squares (GLS)**
   - **Use Case:** When errors have non-constant variance and are correlated with one another, such as in time series data.
   - **Why Use It:** GLS generalizes OLS by allowing for heteroscedasticity and autocorrelation in the errors, providing more efficient estimates in such cases.

### 4. **Ridge Regression (L2 Regularization)**
   - **Use Case:** When there is multicollinearity (high correlation between independent variables), which can lead to unstable OLS estimates.
   - **Why Use It:** Ridge regression adds a penalty term proportional to the square of the magnitude of coefficients, shrinking them toward zero and stabilizing the estimates.

### 5. **Lasso Regression (L1 Regularization)**
   - **Use Case:** Similar to Ridge Regression but especially useful when you want to perform feature selection (i.e., reduce the number of predictors).
   - **Why Use It:** Lasso regression adds a penalty term proportional to the absolute value of the coefficients, which can shrink some coefficients to exactly zero, effectively selecting a simpler model.

### 6. **Elastic Net Regression**
   - **Use Case:** When you need a compromise between Ridge and Lasso regression, combining their penalties.
   - **Why Use It:** Elastic Net is useful when dealing with multiple correlated features, as it can select groups of correlated variables and provide a balance between L1 and L2 regularization.

### 7. **Robust Regression**
   - **Use Case:** When the data contains outliers that would unduly influence OLS estimates.
   - **Why Use It:** Robust regression techniques, such as Huber regression or RANSAC, reduce the influence of outliers, providing estimates that are more resistant to deviations from assumptions.

### 8. **Quantile Regression**
   - **Use Case:** When you want to model the conditional median or other quantiles of the response variable, rather than the mean.
   - **Why Use It:** Quantile regression is useful when the relationship between independent variables and the dependent variable differs across the distribution (e.g., in the tails).

### 9. **Polynomial Regression**
   - **Use Case:** When the relationship between the independent variables and the dependent variable is not linear but can be approximated by a polynomial.
   - **Why Use It:** Polynomial regression models non-linear relationships by including higher-order terms (squares, cubes, etc.) of the independent variables.

### 10. **Principal Component Regression (PCR)**
   - **Use Case:** When there are many correlated predictors, which can lead to multicollinearity.
   - **Why Use It:** PCR reduces dimensionality by performing regression on the principal components, which are linear combinations of the original predictors that capture the most variance.

### 11. **Partial Least Squares (PLS) Regression**
   - **Use Case:** Similar to PCR but when you want to predict the dependent variable while simultaneously considering the variance in both the predictors and the outcome.
   - **Why Use It:** PLS finds components that explain the variance in both X and Y, making it useful when the predictors are highly collinear and when both X and Y have complex structures.

### Choosing the Right Method

- **Data Characteristics:**
  - **Linear vs. Non-linear:** Use OLS for linear relationships; consider polynomial or non-linear models if the relationship isn't linear.
  - **Multicollinearity:** If you have multicollinearity, consider Ridge, Lasso, or Elastic Net regression.
  - **Heteroscedasticity:** Use WLS or GLS if there is non-constant variance in errors.
  - **Outliers:** Consider robust regression if outliers are present.
  - **Feature Selection:** Use Lasso or Elastic Net if you need to reduce the number of predictors.

- **Objective:**
  - **Interpretability vs. Prediction:** OLS and Ridge are typically more interpretable, while Lasso can be used for both prediction and feature selection. Elastic Net is a middle ground.
  - **Handling Complexity:** For complex relationships, use PLS, PCR, or polynomial regression.

Understanding the nature of your data and the assumptions underlying each method will help you choose the appropriate regression technique.

# 1. OLS Statsmodel
Ordinary Least Squares (OLS) is a type of linear regression used in statistics and machine learning to model the relationship between one or more independent variables (predictors) and a dependent variable (outcome). The `statsmodels` library in Python provides a comprehensive framework for performing OLS regression and interpreting the results.

### Key Components of OLS Regression

1. **Dependent Variable (Y):** This is the variable you're trying to predict or explain. In a dataset, it's usually the target variable.

2. **Independent Variables (X):** These are the variables that you use to predict the dependent variable. They are also called predictors or features.

3. **Intercept:** This represents the expected value of the dependent variable when all independent variables are zero. It's the point where the regression line crosses the Y-axis.

4. **coef (Coefficient)**:

     1. This is the estimated value of the regression coefficient for each variable.
     2. const (Intercept): The estimated intercept of the model, which represents the expected value of the dependent variable (Y) when all the predictors (X1, X2) are zero.
       - 0.0333: If both X1 and X2 are zero, the predicted value of Y would be 0.0333.
     3. X1 (0.0500): For each one-unit increase in X1, the dependent variable Y is expected to increase by 0.0500 units, holding X2 constant.
     4. X2 (0.0050): For each one-unit increase in X2, the dependent variable Y is expected to increase by 0.0050 units, holding X1 constant.
5. **std err (Standard Error)**:

     1. This represents the standard error of the estimated coefficient, indicating how much the estimate is expected to vary if the model were to be estimated using a different sample from the same population.
     2. const (0.011): The standard error for the intercept is 0.011, indicating the variability in the intercept estimate.
     3. X1 (0.001): The standard error for the coefficient of X1 is 0.001, suggesting that the estimate is precise (small standard error).
     4. X2 (0.001): Similarly, the standard error for X2 is 0.001.
6. **t (t-Statistic)**:
     1. The t-statistic measures how many standard deviations the coefficient is away from 0. It’s used to test the null hypothesis that the coefficient is equal to zero (no effect).
     2. const (3.033): The intercept has a t-value of 3.033, which means the intercept is 3.033 standard deviations away from zero.
     3. X1 (33.558): The coefficient for X1 has a t-value of 33.558, indicating a very strong relationship between X1 and Y.
     4. X2 (3.856): The coefficient for X2 has a t-value of 3.856, suggesting a significant relationship between X2 and Y, though less strong than X1.
7. **P>|t| (p-value)**:
     1. The p-value indicates the probability of observing the t-statistic (or one more extreme) under the null hypothesis that the coefficient is zero. A low p-value (typically < 0.05) suggests that the     coefficient is statistically significantly different from zero.
     2. const (0.097): The p-value for the intercept is 0.097, which is higher than 0.05, indicating that the intercept is not statistically significant at the 5% level.
     3. X1 (0.001): The p-value for X1 is 0.001, which is very low, suggesting that X1 is statistically significant (it has a strong and significant effect on Y).
     4. X2 (0.059): The p-value for X2 is 0.059, which is slightly above 0.05. This indicates that X2 is not statistically significant at the 5% level, though it is close to significance (it might be significant at the 10% level).
6. **[0.025, 0.975] (95% Confidence Interval)**:

     1. This is the range within which the true value of the coefficient is expected to lie with 95% confidence.
     2. const [-0.012, 0.079]: The confidence interval for the intercept ranges from -0.012 to 0.079. Since this interval includes zero, it suggests that the intercept might not be significantly different from zero.
     3. X1 [0.046, 0.054]: The confidence interval for X1 ranges from 0.046 to 0.054. This narrow interval does not include zero, reinforcing the significance of X1.
     4. X2 [-0.001, 0.011]: The confidence interval for X2 ranges from -0.001 to 0.011. Since this interval includes zero, it suggests that X2 might not be significantly different from zero.
5. **Residuals (Errors):** These are the differences between the observed values of the dependent variable and the values predicted by the model. Tells you how many observations are left after the model has "used up" some data to estimate the coefficients. These remaining observations are what's left to calculate the residuals (the differences between the observed and predicted values).
    - **Low Df Residuals**:
      - If Df Residuals is low, it means most of your data has been used to estimate the model parameters. This can be a sign that the model might be too complex for the amount of data you have. It reduces the reliability of your model because there aren't many data points left to check if the model predictions are accurate.
      - Example: If you have 10 observations and you estimate 8 parameters, you only have 1 Df Residual left. That’s very little data to validate your model, which could lead to overfitting (the model fits the specific data too well but might not generalize to new data).

    - **High Df Residuals**:
      - If Df Residuals is high, you have more data left after estimating the model. This usually means the model is simpler relative to the amount of data, which can be good because it allows for a more reliable check on the model's accuracy.
      - Example: If you have 100 observations and only estimate 3 parameters, you have 97 Df Residuals left, giving you plenty of data to test the model's predictions.

7. **R-squared:** is a statistical measure that represents the proportion of the variance in the dependent variable (Y) that can be explained by the independent variables (X) in the model. It ranges from 0 to 1, where:
    - 0 indicates that the model explains none of the variance.
    - 1 indicates that the model explains all the variance.
    - For example, an R-squared of 0.995 means that 99.5% of the variance in the dependent variable is explained by the independent variables in the model.

8. **Adjusted R-squared:** adjusts the R-squared value for the number of predictors in the model. It is a modified version of R-squared that accounts for the number of independent variables and the sample size. Unlike R-squared, it can decrease if adding more variables doesn’t improve the model enough to justify the increase in complexity.

   **Interpretation of Adjusted R-squared**
   - Why Adjusted R-squared?

      - R-squared always increases when more variables are added to the model, even if those variables do not significantly contribute to the model’s predictive power.
      - Adjusted R-squared accounts for the number of predictors, only increasing if the added variables improve the model more than would be expected by chance. If the added predictors don’t actually improve the model, Adjusted R-squared could decrease.
   - Comparing Models with Different Numbers of Predictors:

      - When comparing models with different numbers of predictors, Adjusted R-squared is a better metric than R-squared because it penalizes the addition of unnecessary variables. A higher Adjusted R-squared indicates a better model when you are considering models with a different number of predictors.
   - **Interpreting Adjusted R-squared with Different Datasets**
      - **Dataset 1**: High R-squared and High Adjusted R-squared

        - Example:
        - R-squared: 0.995
        - Adjusted R-squared: 0.990
      - **Interpretation**: The model explains 99.5% of the variance in the dependent variable. The small drop in Adjusted R-squared indicates that the model is very good, but the complexity (number of predictors) might be slightly more than necessary. The high Adjusted R-squared still confirms that the model is very effective.
    
      - **Dataset 2**: High R-squared but Lower Adjusted R-squared

        - Example:
        - R-squared: 0.900
        - Adjusted R-squared: 0.850
      - **Interpretation**: The model explains 90% of the variance in the dependent variable, but the drop to 85% in Adjusted R-squared suggests that some predictors may not be contributing much value. There might be some overfitting, where unnecessary variables are included, inflating the R-squared.
      - Dataset 3: Low R-squared and Low Adjusted R-squared

        - Example:
        - R-squared: 0.500
        - Adjusted R-squared: 0.400
      - **Interpretation**: The model only explains 50% of the variance, and the Adjusted R-squared further drops to 40%. This suggests that the model is not very effective, and there may be a lot of irrelevant variables, or the relationship between the variables is not well-captured by the model.
      - Dataset 4: High R-squared but No Drop in Adjusted R-squared

        - Example:
        - R-squared: 0.980
        - Adjusted R-squared: 0.979
      - **Interpretation**: The model explains 98% of the variance, and the Adjusted R-squared is almost the same, indicating that most of the predictors are contributing valuable information. This is a sign of a well-fitting model with a good balance between explanatory power and complexity.
10. **p-values:** The p-value associated with the F-statistic (often labeled as Prob (F-statistic)) tells you whether the observed relationship between the dependent variable and the predictors is statistically significant.
    **Understanding the F-statistic and Prob (F-statistic)**
    The F-statistic and its associated p-value (Prob F-statistic) are used to assess the overall significance of a regression model. These metrics help determine whether the model, as a whole, provides a better fit to the data than a model with no independent variables (essentially a model that only includes the intercept).
      - **Interpretation**:

        - The p-value represents the probability that the observed F-statistic (or one more extreme) could occur if the null hypothesis were true. The null hypothesis in this context is that all the regression coefficients (except the intercept) are equal to zero (i.e., the predictors have no effect).

        - A low p-value (typically less than 0.05) indicates that there is a statistically significant relationship between the predictors and the dependent variable.

        - In your example, the Prob (F-statistic) is 0.00477, which is very low. This means there’s less than a 0.477% chance that the relationship observed in your model could be due to random chance. In other words, the model as a whole is statistically significant.

12. **F-statistic:** The F-statistic is a ratio that compares the explained variance of the model (how well the model fits the data) to the unexplained variance (the residuals). In simpler terms, it tests whether at least one of the predictors is significantly related to the dependent variable.

      - **Interpretation**:
        - A higher F-statistic indicates that the model explains a significant portion of the variance in the dependent variable relative to the noise (residual variance).
        - In your example, an F-statistic of 209.0 is quite high, indicating that the model explains much more variance than what would be expected by chance.
        - 
13. **Standard Error:** This measures the average distance that the observed values fall from the regression line. It indicates the precision of the coefficient estimates.

14. **Confidence Intervals:** These provide a range within which the true value of the coefficient is expected to fall with a certain level of confidence (usually 95%).

15. **Omnibus**:  The Omnibus test is a combined test for skewness and kurtosis. It tests the null hypothesis that the residuals (errors) of the model are normally distributed. Specifically, it combines the skewness and kurtosis of the residuals into a single test statistic.
    - **Null Hypothesis (H₀)**: The residuals are normally distributed (i.e., the skewness and kurtosis are both zero).
    - **Interpretation**:
         - A significant Omnibus test (typically indicated by a low p-value, such as < 0.05) suggests that the residuals do not follow a normal distribution.
17. **Prob Omnibus**: This is the p-value associated with the Omnibus test. It tells you the probability of observing the test statistic under the null hypothesis that the residuals are normally distributed.
    - **Interpretation**:
         - A p-value of 0.000 indicates that there is virtually no chance that the residuals are normally distributed. This suggests that the model's residuals deviate significantly from normality, which can be a concern in regression analysis because one of the assumptions of linear regression is that the residuals should be normally distributed.
    - **Omnibus Statistic**: A higher value of the Omnibus statistic indicates a greater deviation from normality.
    - **p-value (Prob(Omnibus))**: The p-value associated with the Omnibus test tells you the probability of observing the test statistic under the null hypothesis.
    - **Ideal Values for the Omnibus Test**:
         - **Omnibus Statistic**:
            - **Near Zero**: Ideally, you want the Omnibus test statistic to be close to zero, which would indicate that the skewness and kurtosis of the residuals are close to what is expected under a normal   distribution. This suggests that the residuals are approximately normally distributed.
         - **Prob(Omnibus) (p-value)**:
            - **High p-value (> 0.05)**: A high p-value (typically above 0.05) suggests that the residuals are not significantly different from a normal distribution. This means you fail to reject the null hypothesis, indicating that the residuals likely follow a normal distribution.
18. **Skewness**: Skewness measures the asymmetry of the distribution of residuals.
    - **Skewness = 0**: Indicates a perfectly symmetrical distribution.
    - **Positive Skew (> 0)**: Indicates that the right tail (larger values) of the distribution is longer or fatter than the left tail, meaning there are more extreme positive residuals.
    - **Negative Skew (< 0)**: Indicates that the left tail (smaller values) is longer or fatter than the right tail, meaning there are more extreme negative residuals.

     **Action**:
      - If the skewness is causing issues in model performance or interpretation, you might consider transformations (e.g., logarithmic transformation) of the dependent variable to reduce skewness.
      - If the skewness is mild (like in this case), it may not require immediate action, but it’s something to keep in mind, especially if other diagnostics suggest model issues.
   
19. **Kurtosis**: Kurtosis measures the "tailedness" of the distribution, indicating the presence of outliers.
    - **Kurtosis = 3**: Indicates a normal distribution with medium tails (mesokurtic).
    - **Kurtosis > 3**: Indicates heavier tails than a normal distribution (leptokurtic), meaning more outliers.
    - **Kurtosis < 3**: Indicates lighter tails than a normal distribution (platykurtic), meaning fewer outliers.
      
    **Action**:
    - No immediate action is typically required when kurtosis is slightly below 3. However, it’s important to continue monitoring other diagnostics to ensure that the overall model fit is adequate.
20. **Durbin-Watson**: The Durbin-Watson (DW) statistic is a test statistic used in regression analysis to detect the presence of autocorrelation (specifically, first-order autocorrelation) in the residuals (errors) of the model.
    **What is Autocorrelation?**

    - **Autocorrelation**: Autocorrelation occurs when the residuals (errors) from a regression model are not independent of each other. In simpler terms, it means that the value of the residual at one time point is correlated with the value of the residual at another time point.
    - **First-Order Autocorrelation**: This specifically refers to the correlation between consecutive residuals.
    - **Durbin-Watson ≈ 2**: No autocorrelation (the residuals are independent).
    - **Durbin-Wtson < 2**: Positive autocorrelation (the residuals are positively correlated).
    - **Durbin-Watson > 2**: Negative autocorrelation (the residuals are negatively correlated).
      
    **When to Take Action**:
   
      - **Durbin-Watson < 1.5**: Indicates potential positive autocorrelation, which may require further investigation or adjustment, such as using time-series models or adding lagged variables.
      - **Durbin-Watson > 2.5**: Indicates potential negative autocorrelation, which might also require investigation, although it's generally less problematic than positive autocorrelation.

21. **Jarque-Bera (JB) Test**: is a statistical test that is used to determine whether the residuals (errors) of a regression model are normally distributed. It specifically assesses the skewness and kurtosis of the residuals to see if they deviate from what is expected under a normal distribution.
    - **Null Hypothesis (H₀)**: The residuals are normally distributed (skewness = 0, kurtosis = 3).
    - **JB Statistic**: The larger the JB statistic, the greater the deviation from normality.
         - **High JB Statistic**: Suggests significant deviation from normality.
         - **Low JB Statistic**: Suggests that the residuals are closer to a normal distribution.
     
22. **p-value (Prob(JB))**: The p-value associated with the JB statistic indicates the probability that the residuals are normally distributed.
    - **High p-value (> 0.05)**: Fail to reject the null hypothesis; the residuals are likely normally distributed.
    - **Low p-value (< 0.05)**: Reject the null hypothesis; the residuals are not normally distributed.
   
**The Omnibus test is a more general test that evaluates overall normality, while the Jarque-Bera test focuses more specifically on matching the skewness and kurtosis of residuals to those of a normal distribution.**

23. **Condition Number (Cond. No.)**: The Condition Number (often abbreviated as Cond. No.) is a diagnostic measure used in regression analysis to assess the sensitivity of the model's predictions to changes in the input data. It is particularly useful for detecting multicollinearity, which occurs when two or more independent variables in the model are highly correlated.
24. 
    **What It Represents**:
      - The Condition Number is a ratio that measures the sensitivity of the output of a model to small changes in the input. In regression, it specifically relates to how much the solution (i.e., the regression coefficients) can change in response to changes in the independent variables.
      - **High Condition Number**: Indicates potential multicollinearity, where small changes in the input data can lead to large changes in the model coefficients. This makes the model unstable and less reliable.
      - **Low Condition Number**: Suggests that the model is stable, and the independent variables are not highly correlated.
      - 
    **Interpreting the Condition Number**:
      - **Cond. No. < 30**: Generally indicates that there is little to no multicollinearity among the independent variables. The model is considered stable.
      - **Cond. No. between 30 and 100**: Suggests moderate multicollinearity, which could lead to some instability in the coefficient estimates.
      - **Cond. No. > 100**: Indicates strong multicollinearity, which could make the model coefficients unreliable and suggest that the independent variables are highly correlated.
        
    **What to Do About High Condition Number**:
    If the Condition Number is high (as in your case), here are some steps you can take to address potential multicollinearity:

    - **Examine the Correlation Matrix**:
         - Check the correlation matrix of the independent variables to identify pairs of variables that are highly correlated.
         - Consider removing or combining highly correlated variables.
           
    - **Variance Inflation Factor (VIF)**:
         - Calculate the Variance Inflation Factor (VIF) for each independent variable. VIF values above 10 typically indicate high multicollinearity.
         - Consider removing variables with high VIFs to reduce multicollinearity.
           
    - **Principal Component Analysis (PCA)**:
         - Apply Principal Component Analysis to reduce the dimensionality of the data, which can help to mitigate multicollinearity by creating new uncorrelated variables (principal components).
           
    - **Regularization Techniques**:
         - Consider using regularization methods like Ridge regression or Lasso regression, which can help reduce the impact of multicollinearity by shrinking the coefficients of correlated variables.
           
    - **Drop Redundant Variables**:
         - If certain variables are redundant or add little new information to the model, consider removing them to improve model stability.
         - 
### Example: Interpreting OLS Results with `statsmodels`

Let's walk through an example using the `statsmodels` library.

```python
import statsmodels.api as sm
import pandas as pd

# Example dataset
data = {
    'Y': [1, 2, 3, 4, 5],
    'X1': [10, 20, 30, 40, 50],
    'X2': [100, 200, 300, 400, 500]
}

df = pd.DataFrame(data)

# Add a constant (intercept) to the model
X = sm.add_constant(df[['X1', 'X2']])
Y = df['Y']

# Fit the OLS model
model = sm.OLS(Y, X).fit()

# Get the summary of the model
summary = model.summary()
print(summary)
```

### Interpreting the Output

Here’s what you might see in the output:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      Y   R-squared:                       0.995
Model:                            OLS   Adj. R-squared:                  0.990
Method:                 Least Squares   F-statistic:                     209.0
Date:                Fri, 17 Aug 2024   Prob (F-statistic):            0.00477
Time:                        12:00:00   Log-Likelihood:                -1.3820
No. Observations:                   5   AIC:                             6.764
Df Residuals:                       2   BIC:                             5.592
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0333      0.011      3.033      0.097      -0.012       0.079
X1             0.0500      0.001     33.558      0.001       0.046       0.054
X2             0.0050      0.001      3.856      0.059      -0.001       0.011
==============================================================================
Omnibus:                        0.996   Durbin-Watson:                   2.450
Prob(Omnibus):                  0.608   Jarque-Bera (JB):                0.444
Skew:                           0.775   Prob(JB):                        0.801
Kurtosis:                       2.484   Cond. No.                     1.61e+03
==============================================================================

```

- **Dep. Variable:** The dependent variable in the model (Y).
- **R-squared (0.995):** Indicates that 99.5% of the variability in Y is explained by the predictors X1 and X2.
- **Adj. R-squared (0.990):** Slightly lower than R-squared, showing the model’s fit adjusted for the number of predictors.
- **F-statistic (209.0) and Prob (F-statistic) (0.00477):** A significant F-statistic (p < 0.05) suggests that the model is statistically significant overall.
- **Coefficients (const, X1, X2):** 
  - **const (0.0333):** The intercept, which is the predicted value of Y when X1 and X2 are zero.
  - **X1 (0.0500):** For each unit increase in X1, Y is expected to increase by 0.05, holding X2 constant.
  - **X2 (0.0050):** For each unit increase in X2, Y is expected to increase by 0.005, holding X1 constant.
- **Standard Error, t, P>|t|, [0.025, 0.975]:** These values show the reliability of the coefficients. For example, X1 has a very low p-value (0.001), making it a significant predictor
- **Df Residuals** tells you how many observations are left after the model has "used up" some data to estimate the coefficients. These remaining observations are what's left to calculate the residuals (the differences between the observed and predicted values).
 
