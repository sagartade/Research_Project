
########## HEADER ##############################################
# Title: R code file related to Financial project
#  
# Description: 
# Ethereum
# 
# Name: Sagar Tade
################################################################

install.packages("Quandl")
library(Quandl)
library(TTR)
library(quantmod)
library(zoo)
require(smooth)
require(Mcomp)
require(tseries) 
require(ggplot2)

bitcoin = Quandl("BITFINEX/ETHUSD", start_date="2010-01-01", end_date="2019-02-28", type = "ts")

# Save to R Dataset file
saveRDS(bitcoin,'./bitcoin.rds')

# Load from R Dataset file
bitcoin <- readRDS('./bitcoin.rds')

ethvar <- rollapply(bitcoin$Last, width = 7, FUN = var, fill = NA)

## Copying timeseries to new variables to keep original values intact
tsdata_bit <- bitcoin$Last

##############Exploration on Ethereum ####################

# Check for stationarity using ADF and KPSS tests
adf.test(tsdata_bit)

kpss.test(tsdata_bit, null="Trend")

decomp_bit <- decompose(ts(na.omit(tsdata_bit), frequency=30))
plot(decomp_bit)
plot(tsdata_bit)

# 
# # Plot the time series data
# autoplot(tsdata_bit, geom = "line") + xlab("Date")+ylab("Megalitres")
# 
# # Find and plot the trend
# tsdata_bit_trend <- ma(tsdata_bit, order = 4, centre = T)
# autoplot(tsdata_bit_trend, geom = "line") + xlab("Date")+ylab("Megalitres")
# 
# tsdata_bit_trend2 <- sma(tsdata_bit, n = 4)
# plot(forecast(tsdata_bit_trend2))
# 
# 
# # Remove the trend and plot the detrended series
# tsdata_bit_detrended <- tsdata_bit - tsdata_bit_trend
# autoplot(tsdata_bit_detrended, geom = "line") + xlab("Date")+ylab("Megalitres")
# 
# 
# # Determine and plot the average seasonality
# tsdata_bit_avg = t(matrix(data = tsdata_bit_detrended, nrow = 4))
# tsdata_bit_seasonal = colMeans(tsdata_bit_avg, na.rm = T)
# autoplot(as.ts(rep(tsdata_bit_seasonal,16)), geom = "line") + xlab("Date") + ylab("Megalitres")
# 
# # Remove the average seasonality and plot the stochastic series
# tsdata_bit_stochastic <- tsdata_bit_detrended - tsdata_bit_seasonal
# autoplot(tsdata_bit_stochastic, geom = "line") + xlab("Date")+ylab("Megalitres")
# 
# 
# # Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the original series
# # and stochastic series
# acf(tsdata_bit)
# pacf(tsdata_bit)
# #acf(tsdata_stochastic)
# acf(na.remove(tsdata_bit_stochastic))
# #pacf(tsdata_stochastic)
# pacf(na.remove(tsdata_bit_stochastic))
# 
# 
# #Attempt to set the model order automatically and print the order of the model
# ar.model_bit <- auto.arima(tsdata_bit)
# arimaorder(ar.model_bit)
# # > arimaorder(ar.model)
# # p         d         q         P         D         Q Frequency 
# # 1         0         2         0         1         1         4 
# 
# 
# # Forecast 8 steps ahead and print/plot the forecast
# fcast <- forecast(ar.model_bit, h = 8)
# print(fcast)
# plot(fcast)
# 
################ Test for checking ARCH ###
library(fDMA)
archtest(ts=tsdata_bit)

library(rugarch)
##Modeling
garch_one_zero <- ugarchspec(variance.model = list(model="sGARCH", garchOrder=c(1,0)),distribution.model = "norm")
garch_one_one <- ugarchspec(variance.model = list(model="sGARCH", garchOrder=c(1,1)),distribution.model = "norm")
gjrgarch_one_zero <- ugarchspec(variance.model = list(model="gjrGARCH", garchOrder=c(1,0)),distribution.model = "norm")
gjrgarch_one_one <- ugarchspec(variance.model = list(model="gjrGARCH", garchOrder=c(1,1)),distribution.model = "norm")
igarch_one_one <- ugarchspec(variance.model = list(model="iGARCH"),distribution.model = "norm")


##Applying GARCH MODELS
garch_bit_one_zero <- ugarchfit(spec=garch_one_zero, data=tsdata_bit)
garch_bit_one_one <- ugarchfit(spec=garch_one_one, data=tsdata_bit)
gjrgarch_bit_one_zero <- ugarchfit(spec=gjrgarch_one_zero, data=tsdata_bit)
gjrgarch_bit_one_one <- ugarchfit(spec=gjrgarch_one_one, data=tsdata_bit)
igarch_bit_one_one <- ugarchfit(spec=igarch_one_one, data=tsdata_bit)
  

##FORECASTING
gjrgarch_bit_one_one_forecast <- ugarchforecast(gjrgarch_bit_one_one, n.ahead = 2, data=tsdata_bit)

plot(gjrgarch_bit_one_one_forecast)
