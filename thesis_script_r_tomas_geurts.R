rm(list = ls())
setwd("C:/.../")

df <- read.csv("final_results_shown_items_total.csv", sep = ";")
df <- df[df$Nr..of.shown.items<250,]

colnames(df) <-
  c('strategy', 'nr_of_items','rmse')

df <- df[order(df$nr_of_items,df$strategy),]

df$nr_of_items <- as.factor(df$nr_of_items)

require(lattice)


xyplot(
  rmse ~ nr_of_items,
  type = 'b',
  groups = strategy,
  data = df,
  ylim=range(0.3,0.525),
  xlab = "Number of items shown to the cold users",
  ylab = "RMSE",
  col = c("orange","blue","yellow","green","red","purple"),
  key = list(
    space = "right",
    lines =
      list(
        col = c("purple","red","blue","orange","green","yellow")
      ),
    text =
      list(c("Random strategy","Popularity strategy","Gini strategy","Entropy strategy","PopGini strategy","PopEnt strategy"))
  )
)