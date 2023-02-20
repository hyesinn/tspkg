library(forecast)


#info <- list(
#  trend_num = 4, 
#  y = "Beer",
#  features = "Trend/Season"
#)
#result <- read.csv("C:/Users/teakyun kim/Documents/GitHub/tspkg/Result/result.csv", header = FALSE )
#data <- read.csv("C:/Users/teakyun kim/Documents/GitHub/tspkg/Data/aus_production.csv", stringsAsFactors = FALSE)

data$trend  <- seq(1,nrow(data))
season      <- rep(1:info$trend_num,nrow(data)/info$trend_num+1)
data$season <- as.factor(season[1:nrow(data)])


if(info$features == "Trend"){
  
  
  
  lm.fit <- lm(paste0(info$y,"~","trend"), data = data)
  
  
}else if(info$features == "Season"){
  
  lm.fit <- lm(paste0(info$y,"~","season"), data = data)
 
}else if(info$features == "Trend/Season"){
  
  lm.fit <- lm(paste0(info$y,"~","season","+","trend"), data = data)
  
}else{
  print("Incorrect feature is typed. Please select feature one of follows : Trend, Season, Trend/Season.")
}

print("validation result as below : ")
print("compare result round 3 decimals")
print(table(round(lm.fit$fitted.values,3) == round(result,3)))
print("compare result round 4 decimals")
print(table(round(lm.fit$fitted.values,4) == round(result,4)))
print("compare result round 5 decimals")
print(table(round(lm.fit$fitted.values,4) == round(result,5)))