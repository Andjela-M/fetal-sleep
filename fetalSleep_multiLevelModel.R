# assess relationship between fetal and maternal signals
# SLOOP - Andjela Markovic, December 2023

library(tidyverse)
library(lmerTest)
library(lubridate)
library(performance)
library(sjPlot)
library(sjmisc)
library(ggplot2)
library(emmeans)
library(ggeffects)

# load data
load("FetalData.RData")

# run models with fetal activity as outcome nested within days
allData$subject <- as.factor(allData$subject)
allData$state <- as.factor(allData$state)
allData$twins <- as.factor(allData$twins)
allData$ivf <- as.factor(allData$ivf)
allData$diabetes <- as.factor(allData$diabetes)
allData$nulliparity <- as.factor(allData$nulliparity)

# define dataset
intModel <- lmer(bellyAbsZ ~ state * age * watchtemperatureZ + state * age * watchZ + twins + ivf + diabetes + nulliparity + (day | subject),control=lme4::lmerControl(optimizer="bobyqa"),
                 data = allData, REML = TRUE)
dataSubset <- na.omit(allData[ , all.vars(formula(intModel))])

# null model with random intercept
nullModel <- lmer(bellyAbsZ ~  (1 | subject),control=lme4::lmerControl(optimizer="bobyqa"),
                 data = dataSubset, REML = TRUE)
summary(nullModel)
AIC(nullModel)
performance(nullModel)

# fixed factors
fixedModel <- lmer(bellyAbsZ ~ age + state + watchtemperatureZ + watchZ + twins + ivf + diabetes + nulliparity + (1 | subject),control=lme4::lmerControl(optimizer="bobyqa"),
                   data = dataSubset, REML = TRUE)
summary(fixedModel)
AIC(fixedModel)
performance(fixedModel)

# random intercept and slope (intercept 1 automatically included as in (1+day|subject))
randomModel <- lmer(bellyAbsZ ~ age + state + watchtemperatureZ + watchZ + twins + ivf + diabetes + nulliparity + (day | subject),control=lme4::lmerControl(optimizer="bobyqa"),
                   data = dataSubset, REML = TRUE)
summary(randomModel)
AIC(randomModel)
performance(randomModel)

# interaction model
intModel <- lmer(bellyAbsZ ~ state * age * watchtemperatureZ + state * age * watchZ + twins + ivf + diabetes + nulliparity + (day | subject),control=lme4::lmerControl(optimizer="bobyqa"),
                 data = dataSubset, REML = TRUE)

summary(intModel)
AIC(intModel)
performance(intModel)