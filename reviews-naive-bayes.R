library(tm)
library(RTextTools)
library(e1071)
library(dplyr)
library(caret)

df <- read.csv("Training data", stringsAsFactors = FALSE)
df.test <- read.csv("Test data", stringsAsFactors = FALSE)
View(df)
df$rating <- as.factor(df$rating)
df.test$rating <- as.factor(df.test$rating)

corpus <- Corpus(VectorSource(df$review))
corpus.test <- Corpus(VectorSource(df.test$review))
corpus
inspect(corpus[1:3])

#CORPUS CLEANUP
corpus.clean <- corpus %>% tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

corpus.test.clean <- corpus.test %>% tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)


#DOCUMENT TERM MATRIX
dtm <- DocumentTermMatrix(corpus.clean)
dtm



dtm.test <- DocumentTermMatrix(corpus.test.clean)


inspect(dtm[40:50, 10:15])

#FEATURE SELECTION
frequency <- findFreqTerms(dtm, 7)
frequency.test <- findFreqTerms(dtm.test, 5)
dtm <- DocumentTermMatrix(corpus.clean, control=list(dictionary = frequency))

dtm.test <- DocumentTermMatrix(corpus.test.clean, control=list(dictionary = frequency.test))





#Convert word frequencies to yes and no
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}

dtm <- apply(dtm, 2, convert_count)
dtm.test <- apply(dtm.test, 2, convert_count)

dtm<-as.data.frame(as.matrix(dtm))
dtm.test<-as.data.frame(as.matrix(dtm.test))

#NAIVE BAYES
naive.model <- naiveBayes(dtm, df$rating, laplace = 1)
pred <- predict(naive.model, newdata=dtm.test)

#Prediction table
table("Predictions"= pred,  "Actual" = df.test$rating )

#COnfusion matrix
conf.mat <- confusionMatrix(pred, df.test$rating)
conf.mat$overall['Accuracy'] * 100
