library(dplyr)
library(tidyr)
library(ggplot2)
library(tabplot)
source('multiplot.R')

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

# Read data
df <- read.csv('../data/train.csv')

# Transform response variable, SalePrice
df <- mutate(df, logSalePrice = log(SalePrice)) %>%
  select(-SalePrice)
response = 'logSalePrice'

# Remove ID
ID = df['Id']
df <- select(df,-Id)

# Keep only first 20 columns plus response

# Model Baseline


# MSSubClass
# Change to factor
df <- mutate(df, MSSubClass = factor(MSSubClass))

#LotFrontage
# Add squared term
df <- mutate(df,sqLotFrontage = LotFrontage^2)

# LotArea
# Use sqrt of LotArea
df <- mutate(df,sqrtLotArea = LotArea^(1/2))
generate_scatter_plot('sqrtLotArea')
# Look at benefit of having LotArea + sqLotArea --> keep both
lm_LotArea <- lm(logSalePrice ~ sqrtLotArea + LotArea, data=df)
anova(lm_LotArea)

# YearBuilt
# Add squared term
df <- mutate(df,sqYearBuilt = YearBuilt^2)


# OverallQual and HouseStyle
#Look for interactions between OverallQual*HouseStyle
lm_interactions1 <- lm(logSalePrice ~ HouseStyle*OverallQual,
                       data=df)
summary(lm_interactions1)
# In Python, add interaction for HouseStyle1Story:OverallQual and HouseStyle2Story:OverallQual

# Overall condition 
lm_condition1 <- lm(logSalePrice ~ OverallCond, data=df)
summary(lm_condition1)

# change to very bad, bad, neutral, good, very good
df <- df %>%
  mutate(VeryBadCondition = ifelse(OverallCond<=2,1,0)) %>%
  mutate(BadCondition = ifelse(OverallCond<=4&OverallCond>2,1,0)) %>%
  mutate(NeutralCondition = ifelse(OverallCond<=6&OverallCond>4,1,0)) %>%
  mutate(GoodCondition = ifelse(OverallCond<=8&OverallCond>6,1,0)) %>%
  mutate(VeryGoodCondition = ifelse(OverallCond>8,1,0))

lm_condition2 <- lm(logSalePrice ~ VeryBadCondition + BadCondition +
                      NeutralCondition + GoodCondition + VeryGoodCondition, data=df)
summary(lm_condition2)

# Remove overall condition
df <- select(df, -OverallCond)

# Condition 1 and 2
lm_proximity1 = lm(logSalePrice ~ Condition1 + Condition2, data=df)
summary(lm_proximity1)
# Change to: near major road,near positive, near RR
df <- df %>%
  mutate(NearMajorRoad = ifelse(Condition1%in%c('Artery','Feedr')|Condition2%in%c('Artery','Feedr'),1,0))%>%
  mutate(NearPositive = ifelse(Condition1%in%c('PosA','PosN')|Condition2%in%c('PosA','PosN'),1,0))%>%
lm_proximity2 = lm(logSalePrice ~ NearMajorRoad + NearPositive , data=df)
summary(lm_proximity2)
# Remove condition 1, condition2
df <- select(df, -Condition1)
df <- select(df, -Condition2)

# LotShape
# Change to irregular or not
df <- df %>%
  mutate(LotShapeIR = ifelse(LotShape%in%c('IR1','IR2','IR3'),1,0)) %>%
  select(-LotShape)

#HouseStyle
# Give ordering based on number of floors
df<-df %>% 
  mutate(NumFloors = as.character(HouseStyle)) %>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='1.5Unf', '1'))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='1Story', NA))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='1.5Fin', '3'))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='SFoyer', '4'))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='SLvl', '5'))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='2.5Unf', '6'))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='2Story', '7'))%>%
  mutate(NumFloors = replace(NumFloors, HouseStyle=='2.5Fin', '8'))%>%
  mutate(NumFloors = as.numeric(NumFloors))
  
generate_scatter_plot('HouseStyle')
generate_scatter_plot('NumFloors')

  
# Imputation:
# For categorical replace "nothing with NA"
# Many of the square footage = 0 should be replaced with NA

# Ideas:
# Month Sold to Season
# Remove categories with less than 10 points
# Order categories by quality where possible



