library(readr)
library(tsibble)
library(feasts)
library(dplyr)
library(ggplot2)

pwd <- getwd()
DIR_out <- sprintf("%s/results_structural_change_Sep2023/studyIV/", pwd)
timescaless <- list("daily", "hourly") 
commodities <- list("corn", "wheat")
contract <- list("front", "second")
mprocesses <- list("PriceRaw", "PriceReturns", "volume", "RealVolatility")
df <- read_csv(sprintf("%s/results_structural_change_Sep2023/studyIV/studyIV_data.csv", pwd))
for (comm in commodities){
   for (contr in contract){
         if (comm == "wheat"){
            rname <- list("PP", "ACR", "CPAS", "GS", "SGAS", "CPSS", "WWS", "CPM", "WASDE")
         } else if (comm == "corn") {
            rname <- list("PP", "ACR", "CPAS", "WASDE", "CPSS", "GS")
         }
         for (timescale in timescaless){
                  lags <- switch(timescale,
                                 "daily" = list(1, 2, 3, 4, 5),
                                 "minute" = list(15, 30, 45, 60),
                                 "hourly" = list(1, 4)
                                 )
                  for (lag in lags){
                     for (mprocess in mprocesses){                           
                        # for series plots
                        #    plotdf <- df %>% filter(commodity == comm,
                        #                            marketproc == mprocess,
                        #                            timescales == timescale,
                        #                            lagstruct == lag,
                        #                            contracts == contr,
                        #                            ) %>% 
                        #                            mutate(ym = yearmonth(as.character(as.Date(reportdate, 
                        #                                  format = "%d %Y %b")))) %>%
                        #                            select(-reportdate) %>%
                        #                            arrange(ym) %>%
                        #                            group_by(ym) %>%
                        #                            summarise(oneminusp = quantile(oneminusp, p = 0.51, type = 1)) %>% 
                        #                            as_tsibble(index = ym)
                        #    plotdf <- fill_gaps(plotdf)

                        #    if (nrow(plotdf) == 0){
                        #          print(mprocess)
                        #          print(comm)
                        #          print(lag)
                        #          print(contr)
                        #          next
                        #    }
                        
                        #    jpeg(sprintf("%ssubseries_%s_%s_%s_%s_%s.jpeg", DIR_out, comm, contr, lag, mprocess, timescale), width=1024)                   
                        #    pr <- plotdf %>% gg_subseriesmedian(oneminusp) + ylab("1 - pvalue")
                        #    print(pr)
                        #    dev.off()

                           # pdf(sprintf("%sseason_%s_%s_%s_%s_%s_%s_%s.pdf", DIR_out, comm, contr, lex, lag, mprocess, caus, mdir))               
                           # plotdf %>% gg_season(oneminusp, polar = TRUE) + ylab("1 - pvalue") + xlab("Month")
                           # dev.off()                  
                           # boxplots
                           plotdf <- df %>% filter(commodity == comm,
                                                   marketproc == mprocess, 
                                                   lagstruct == lag,
                                                   contracts == contr,
                                                   timescales == timescale) %>%                            
                                    arrange(year) %>%
                                    mutate(m = months(as.Date(reportdate, 
                                                format = "%d %Y %b"))) %>%
                                    arrange(m)
                           if (nrow(plotdf) == 0){
                                 print(mprocess)
                                 print(comm)
                                 print(lag)
                                 print(contr)
                                 print("CHECK")
                                 next
                           }                          

                           plotdf$m <- factor(plotdf$m, levels=c("January", "February", 
                            "March", "April", "May", "June", "July", "August", "September", "October", "November",
                            "December"))
                           

                           jpeg(sprintf("%sboxmonth_%s_%s_%s_%s_%s.jpeg", DIR_out, comm, contr, lag, mprocess, timescale), width=1024)                                             
                           p <- ggplot(data=plotdf, aes(x=factor(m), y=oneminusp, fill=factor(farmingperiod))) + 
                              geom_boxplot(position=position_dodge(1)) + ylim(0, 1) + ylab("1 - pvalue") + xlab("month") +
                              theme(text = element_text(size = 20), legend.text = element_text(size=20), legend.title=element_blank()) + 
                              facet_wrap(~m, scale="free", ncol=12)
                           print(p)
                           dev.off()
                        }
                  }
         }
   }
}