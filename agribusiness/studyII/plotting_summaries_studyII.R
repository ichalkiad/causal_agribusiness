library(readr)
library(tsibble)
library(feasts)
library(dplyr)
library(ggplot2)
library(forecast)

commodities <- list("corn") #"wheat"
contract <- list("front", "second")
dictionaries <- list("customdict", "usdacmecftc")
lags <- list(1, 3, 5)
mprocesses <- list("price", "priceret", "vol", "realvol", "volrealvol")
causaldir <- list("mean", "meancov")
mdirection <- list("m2sent", "sent2m")
pwd <- getwd()

for (comm in commodities){
   for (contr in contract){
      for (lex in dictionaries){
         for (lag in lags){
            for (mprocess in mprocesses){
               for (caus in causaldir){
                  if (caus == "meancov" & lag == 5){
                     next
                  }
                  for (mdir in mdirection){
                     df <- read_csv(sprintf("%s/results_out_causality_Sep2023/studyII/%s.csv",  pwd, comm))
                     plotdf <- df %>% filter(marketprocess == mprocess,
                                    msentdirection == mdir,
                                    lagval == lag,
                                    causalloc == caus,
                                    contractval == contr,
                                    lexicon == lex) %>%
                              mutate(ym = yearmonth(as.character(as.Date(datesval,
                                    format = "%d %Y %b")))) %>%
                              select(-datesval) %>%
                              arrange(ym) %>%
                              group_by(ym) %>%
                              summarise(oneminusp = mean(oneminusp)) %>%
                              as_tsibble(index = ym)
                     plotdf <- fill_gaps(plotdf)
                     DIR_out <- sprintf("%s/results_out_causality_Sep2023/studyII/subseries/jpeg/median/", pwd)
                     # pdf(sprintf("%ssubseries_%s_%s_%s_%s_%s_%s_%s.pdf",
                     #  DIR_out, comm, contr, lex, lag, mprocess, caus, mdir))               
                     # plotdf %>% gg_subseries(oneminusp) + ylab("1 - pvalue")
                     # dev.off()                  
                     jpeg(sprintf("%ssubseries_%s_%s_%s_%s_%s_%s_%s.jpeg", DIR_out, comm, contr, lex, lag, mprocess, caus, mdir), width = 1024)                     
                     pr <- plotdf %>% gg_subseriesmedian(oneminusp) +
                           ylab("1 - pvalue") + ylim(0, 1) +
                           xlab("Year - month")
                           theme(text = element_text(size = 20),
                           legend.text = element_text(size = 20),
                           legend.title = element_blank(), 
                           axis.text.x = element_text(angle = 90,
                           vjust = 0.5, hjust = 1))                           
                     print(pr)
                     dev.off()
               }
            }
         }
      }
   }
}
}


