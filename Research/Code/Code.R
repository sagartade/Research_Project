
########## HEADER ##############################################
# Title: R code file related to Financial project
#  
# Description: 
# Ethereum
# 
# Name: Sagar Tade
################################################################

#install.packages("Quandl")
library(readxl)
#library(Quandl)
library(TTR)
library(quantmod)
library(zoo)
require(smooth)
require(Mcomp)
require(tseries) 
require(ggplot2)
library(plotly)
#Bitcoin = Quandl("BITFINEX/ETHUSD", start_date="2010-01-01", end_date="2019-07-28", type = "ts")

#Data taken from Coin base as it had more records than Quandl
Ethereum <- read_excel("D:/Sagar/Study/Sem3/Research_Thesis/Research_Project/Research/Code/Dataset/Ethereum.xlsx")

# Save to R Dataset file
saveRDS(Ethereum,'./ethereum.rds')

# Load from R Dataset file
Ethereum <- readRDS('./ethereum.rds')

Ethereum <- Ethereum[order(as.Date(Ethereum$Date)),] #Ordered in Ascending dates

eth <- Ethereum

##############Exploration on Ethereum ####################

# Plotting Ethereum Prices in Candlestick
# Reference https://plot.ly/r/candlestick-charts/
eth_plot <- eth %>%
  plot_ly(x = ~Date, type="candlestick",
          open = eth$Open, close = eth$Close,
          high = eth$High, low = eth$Low) %>%
  layout(title = "Ethereum Price Chart", xaxis = list(rangeslider = list(visible = F)))

eth_plot

##Decompose
tsdata_bit <- eth$Close
decomp_bit <- decompose(ts(na.omit(tsdata_bit), frequency=30))
plot(decomp_bit)

#Transforming prices to log for feeding the model
a = log(eth$Close)
log_price <- a
eth <- cbind(eth, log_price)

# Difference between log prices
roc <- ROC(eth[,"Close"])
pct_change <- roc
eth <- cbind(eth, pct_change)

# Standard Deviation of

library("caTools")
c <- xts(x=pct_change, order.by = eth$Date)
b = roll_sd(c, 30, center = F)
d <- data.frame(date=index(b), coredata(b))
stdev <- d$coredata.b
eth <- cbind(eth, stdev)


#Rolling Volatility With 30 Time Periods
Volatility <- volatility(eth[-1], n = 30, calc = "yang.zhang", N = 365)

eth <- cbind(eth, Volatility) #Appending volatility to original dataframe
plot_ly(x = eth$Date, y = volatility, type = 'scatter', mode = 'lines') %>%
          layout(title = "Rolling Volatility With 30 Time Periods - Yang Zang Estimator",
                 xaxis = list(title = "Index"),
                 yaxis = list (title = "Volatility"))



# Check for stationarity using ADF and KPSS tests
adf.test(na.omit(volatility))
kpss.test(na.omit(volatility), null="Trend")

# Test for checking ARCH ###
library(fDMA)
archtest(ts=volatility)

# Exporting data to excel
l <- list(df = eth)
openxlsx::write.xlsx(l, file = "D:\\Sagar\\Study\\Sem3\\Research_Thesis\\Research_Project\\Research\\Code\\Dataset\\Updated_Ethereum.xlsx")





###################### GARCH Modelling on Volatility #########################################

# library(rugarch)
# ##Modeling
# garch_one_zero <- ugarchspec(variance.model = list(model="sGARCH", garchOrder=c(1,0)),distribution.model = "norm")
# garch_one_one <- ugarchspec(variance.model = list(model="sGARCH", garchOrder=c(1,1)),distribution.model = "norm")
# gjrgarch_one_zero <- ugarchspec(variance.model = list(model="gjrGARCH", garchOrder=c(1,0)),distribution.model = "norm")
# gjrgarch_one_one <- ugarchspec(variance.model = list(model="gjrGARCH", garchOrder=c(1,1)),distribution.model = "norm")
# igarch_one_one <- ugarchspec(variance.model = list(model="iGARCH"),distribution.model = "norm")
# 
# 
# ##Applying GARCH MODELS
# garch_bit_one_zero <- ugarchfit(spec=garch_one_zero, data=tsdata_bit)
# garch_bit_one_one <- ugarchfit(spec=garch_one_one, data=tsdata_bit)
# gjrgarch_bit_one_zero <- ugarchfit(spec=gjrgarch_one_zero, data=tsdata_bit)
# gjrgarch_bit_one_one <- ugarchfit(spec=gjrgarch_one_one, data=tsdata_bit)
# igarch_bit_one_one <- ugarchfit(spec=igarch_one_one, data=tsdata_bit)
#   
# 
# ##FORECASTING
# gjrgarch_bit_one_one_forecast <- ugarchforecast(gjrgarch_bit_one_one, n.ahead = 2, data=tsdata_bit)

# plot(gjrgarch_bit_one_one_forecast)
