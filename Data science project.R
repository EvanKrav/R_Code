library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(randomForest)
library(xgboost)
library(readxl)
library(scales)
library(Metrics)
library(reshape2)
library(plotly)

AmesHousing <- read_excel("C:/Users/evang/Desktop/r4ds/AmesHousing.xlsx")
View(AmesHousing)
summary(AmesHousing)
str(AmesHousing)

#Changes Lot Frontage rows that have missing values into the calculated Median 
AmesHousing$`Lot Frontage`[is.na(AmesHousing$`Lot Frontage`)] <- median(AmesHousing$`Lot Frontage`, na.rm = TRUE)

#Drop rows with remaining NA values
AmesHousing<- na.omit(AmesHousing)

ggplot(AmesHousing, aes(x=SalePrice))+
  geom_histogram(bins=50, fill="Blue", color = "Black")+
  labs(title = "Distribution of House Sale Prices", x = "Sale Price", y = "Count")


ggplot(AmesHousing, aes(x=Neighborhood, y = SalePrice))+
  geom_boxplot()+
  theme(axis.text.x = element_text(angle=90,hjust = 1))+
  labs(title = "Sale Price by Neighborhood", x="Neighborhood", y ="Sale Price")+
  scale_y_continuous(labels= comma)

AmesHousing$'Total Square Footage' <- AmesHousing$`Gr Liv Area`+AmesHousing$`Total Bsmt SF`

AmesHousing$SalePrice <- log(AmesHousing$SalePrice)


#Splitting the data 80/20 for testing/training
set.seed(123)
trainIndex <- createDataPartition(AmesHousing$SalePrice, p=0.8,list=FALSE, times = 1)

house_train <- AmesHousing[trainIndex,]
house_test <- AmesHousing[-trainIndex,]

#Training a Basic Linear Regression model
lm_model <- train(SalePrice ~ ., data= AmesHousing, method ="lm")
summary(lm_model)

#Predictions and Evaluation
predictions <- predict(lm_model, house_test)
RMSE(predictions, house_test$SalePrice)

# Calculate RMSE and R2
rmse <- RMSE(predictions, house_test$SalePrice)
r_squared <- R2(predictions, house_test$SalePrice)

# Print metrics
print(paste("RMSE: ", round(rmse, 2)))
print(paste("R²: ", round(r_squared, 2)))


correlation_matrix <- cor(house_train[, sapply(house_train, is.numeric)])

correlation_melted <- melt(correlation_matrix)

# Step 3: Create the heatmap
ggplot(data = correlation_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +  # Add white borders to tiles
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), 
                       name = "Correlation") +  # Color gradient
  theme_minimal() +  # Clean theme
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels
  labs(title = "Correlation Matrix Heatmap", x = "", y = "")  # Add title and labels


model <- lm(AmesHousing$SalePrice ~ AmesHousing$`Year Built`, data=AmesHousing)

ggplot(AmesHousing, aes(x=AmesHousing$`Year Built`, y=AmesHousing$SalePrice))+
  geom_point()+
  geom_smooth(method= "lm", se= FALSE, color= "pink")+
  labs(title="Linear Regression: Sale Price vs Year Built",x ="Year Built", y= "Sale Price")
  

model2 <- lm(AmesHousing$SalePrice ~ AmesHousing$`Year Built`+AmesHousing$`Year Remod/Add`, data=AmesHousing)

plot_ly(AmesHousing, x=AmesHousing$`Year Built`, y =AmesHousing$`Year Remod/Add`, z = AmesHousing$SalePrice, type="scatter3d",mode="markers",color = "black")%>%
  layout(title= "3D Scatter Plot", scene= list(xaxis = list(title="Year Built"),yaxis = list(title= "Year Remodeled"),zaxis= list(title= "Sale Price")))




################################################################################################
library(shiny)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(plotly)
library(reshape2)



ui <- fluidPage(
  titlePanel("Ames Housing Data Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("neighborhood", "Select Neighborhood:", 
                  choices = unique(AmesHousing$Neighborhood), 
                  selected = "NAmes", multiple = TRUE),
      sliderInput("num_trees", "Number of Trees (Random Forest):", min = 10, max = 500, value = 100, step = 10)
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Distribution", 
                 plotOutput("dist_plot"),
                 plotOutput("box_plot")),
        tabPanel("Correlation Heatmap", 
                 plotOutput("heatmap_plot")),
        tabPanel("3D Scatter Plot", 
                 plotlyOutput("scatter3d_plot")),
        tabPanel("Model Evaluation", 
                 tableOutput("model_metrics"),
                 plotOutput("feature_importance"))
      )
    )
  )
)

# Server
server <- function(input, output) {
  # Filtered Data
  filtered_data <- reactive({
    AmesHousing %>% filter(Neighborhood %in% input$neighborhood)
  })
  
  # Distribution Plot
  output$dist_plot <- renderPlot({
    ggplot(filtered_data(), aes(x = SalePrice)) +
      geom_histogram(bins = 50, fill = "blue", color = "black") +
      labs(title = "Sale Price Distribution", x = "Sale Price", y = "Count")
  })
  
  # Box Plot
  output$box_plot <- renderPlot({
    ggplot(filtered_data(), aes(x = Neighborhood, y = SalePrice)) +
      geom_boxplot() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
      labs(title = "Sale Price by Neighborhood", x = "Neighborhood", y = "Sale Price")
  })
  
  # Correlation Heatmap
  output$heatmap_plot <- renderPlot({
    correlation_matrix <- cor(filtered_data() %>% select_if(is.numeric))
    correlation_melted <- melt(correlation_matrix)
    ggplot(correlation_melted, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile(color = "white") +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                           midpoint = 0, limit = c(-1, 1), name = "Correlation") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      labs(title = "Correlation Matrix Heatmap")
  })
  
  # 3D Scatter Plot
  output$scatter3d_plot <- renderPlotly({
    plot_ly(filtered_data(), x = ~`Year Built`, y = ~`Year Remod/Add`, z = ~SalePrice, 
            type = "scatter3d", mode = "markers") %>%
      layout(scene = list(xaxis = list(title = "Year Built"),
                          yaxis = list(title = "Year Remodeled"),
                          zaxis = list(title = "Sale Price")))
  })
  
  # Random Forest Model
  rf_model <- reactive({
    train(SalePrice ~ ., data = filtered_data(), method = "rf", 
          trControl = trainControl(method = "cv", number = 5),
          tuneGrid = expand.grid(mtry = sqrt(ncol(filtered_data()))),
          ntree = input$num_trees)
  })
  
  # Model Metrics
  output$model_metrics <- renderTable({
    rf_predictions <- predict(rf_model(), newdata = filtered_data())
    data.frame(
      RMSE = RMSE(rf_predictions, filtered_data()$SalePrice),
      R2 = R2(rf_predictions, filtered_data()$SalePrice)
    )
  })
  
  # Feature Importance
  output$feature_importance <- renderPlot({
    importance <- varImp(rf_model(), scale = TRUE)$importance
    importance <- as.data.frame(importance)
    importance$Feature <- rownames(importance)
    ggplot(importance, aes(x = reorder(Feature, Overall), y = Overall)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(title = "Feature Importance (Random Forest)", x = "Features", y = "Importance")
  })
}

# Run App
shinyApp(ui = ui, server = server)
