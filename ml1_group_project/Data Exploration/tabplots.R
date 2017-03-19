library(tabplot)
library(data.table)
library(dplyr)
library(tidyr)

# Read data
df <- read.csv('../data/train.csv')

# Transform response variable, SalePrice
df <- mutate(df, logSalePrice = log(SalePrice)) %>%
  select(-SalePrice)

for (i in 1:80) {
  if (typeof(df[, i]) == "factor") {
    df[is.na(df[, i]), i] <- ""
    df[, i] <- as.factor(df[, i])
  }
}

for (i in 1:20) {
  png(paste(c('plots/taplot',i,'.png'),collapse=''), width = 6, height = 4, units = 'in', res = 300)
  plot(tableplot(df, 
                 select = c(((i - 1) * 4 + 1):(i * 4), 81), 
                 sortCol = 5, 
                 nBins = 73, 
                 plot = FALSE),
       fontsize = 12,
       title = paste("log(SalePrice) vs ", paste(colnames(df)[((i - 1) * 4 + 1):(i * 4)], collapse = "+"), sep = ""),
       showTitle = TRUE,
       fontsize.title = 12)
  dev.off()
}

