library(dplyr)
library(tidyr)
library(ggplot2)
source('multiplot.R')

# Read data
df <- read.csv('../data/train.csv')

# Transform response variable, SalePrice
df <- mutate(df, logSalePrice = log(SalePrice)) %>%
  select(-SalePrice)

# Get variable names of predictors/response
predictors = colnames(df)[1:80]
response = colnames(df)[81]

# Define function to create scatter plots
generate_scatter_plot <- function(predictor){
  # Scatter plot y against x with OLS line and LOESS line
  if (is.factor(df[,predictor])) {
    p <- df %>%
      ggplot(aes_string(x=predictor, y=response)) +
      geom_point() + 
      geom_boxplot() +
      theme(text = element_text(size=8))
  } else {
    p <- df %>%
      ggplot(aes_string(x=predictor, y=response)) +
      geom_point() + 
      geom_smooth(method='lm',color='blue') +
      geom_smooth(method='loess', color='red') +
      theme(text = element_text(size=8))
    return(p)
  }
}

# Generate scatter plots for all predictors
for (i in 1:20){
  png(paste(c('plots/scatterplot',i,'.png'),collapse=''), width = 8, height = 4, units = 'in', res = 300)
  predictors_to_plot = predictors[c(((i - 1) * 4 + 1):(i * 4))]
  p1 <- generate_scatter_plot(predictors_to_plot[1])
  p2 <- generate_scatter_plot(predictors_to_plot[2])
  p3 <- generate_scatter_plot(predictors_to_plot[3])
  p4 <- generate_scatter_plot(predictors_to_plot[4])
  multiplot(p1, p2, p3, p4, cols=2)
  dev.off()
}
