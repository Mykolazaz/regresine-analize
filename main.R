library(readr)

adult <- read_csv("./data/adult.data", col_names = FALSE)
colnames(adult) <- c("age", "workclass", "fnlwgt", "education", "education_num",
                     "marital_status", "occupation", "relationship", "race",
                     "sex", "capital_gain", "capital_loss", "hours_per_week",
                     "native_country", "income")

adult$income <- ifelse(adult$income == "<=50K", 0, 1)
