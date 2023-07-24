# lbirary import
library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(randomForest)
library(caret)
library(ROCR)
library(ComplexHeatmap)



# data loading
data <- read_delim("/nfs-data/ULMS/GTEx_TCGA_TARGET/data/TcgaTargetGtex_rsem_gene_fpkm", 
                   "\t", escape_double = FALSE, trim_ws = TRUE)
phenotype <- read.csv('/nfs-data/ULMS/GTEx_TCGA_TARGET/data/TcgaTargetGTEX_phenotype.csv',header = TRUE)
ULMS_list <- read.csv("/nfs-data/ULMS/GTEx_TCGA_TARGET/data/TcgaTargetGtex_ulms_list.csv",header = TRUE)
Uterus_list <- read.csv("/nfs-data/ULMS/GTEx_TCGA_TARGET/data/TcgaTargetGTEX_Uterus_list.csv",header = TRUE)
gene_data <- read_delim('/nfs-data/ULMS/data/gencode.v23.annotation.gene.txt', '\t')
inter_gene <- read.csv('/nfs-data/ULMS/pathway_model/data/ref/TCGA_theragen_inter_gene.csv', header=TRUE, stringsAsFactors=FALSE)

data$sample <- substr(data$sample,1,15)
gene_data$id <- substr(gene_data$id,1,15)


# TCGA & theragen gene intersection & select uterus samples
inter_gene_data <- data[data$sample %in% inter_gene$sample,]
rownames(inter_gene_data) <- NULL
uterus_data <- cbind(sample=inter_gene_data[1], inter_gene_data[,colnames(inter_gene_data) %in% Uterus_list$sample])

# 0. GAPDH scale
uterus_data_T <- uterus_data %>% column_to_rownames('sample') %>% t() %>% as.data.frame()  %>% rownames_to_column('sample')
'ENSG00000111640' %in% colnames(uterus_data_T)
gapdh_value <- uterus_data_T['ENSG00000111640']$ENSG00000111640

head(uterus_data_T['ENSG00000111640']) # scale 전
uterus_data_T <- cbind(sample=uterus_data_T[,c(1)],
                       uterus_data_T[,-c(1)] %>% apply(2, function(x){x - gapdh_value}) %>% as.data.frame())
head(uterus_data_T['ENSG00000111640']) # scale 후
summary(uterus_data_T[1:5, 1:5])

uterus_data_T <- read.csv('/nfs-data/ULMS/Diane/data/GAPDH_scale_inter_gene_uterus_data.csv', header=TRUE, stringsAsFactors=FALSE)

# train test split
tr_sample <- read.csv('/home/yjseo/ulms_2021/ulms_list/data/TCGA_train_samples_2.csv', header=TRUE, stringsAsFactors=FALSE)
ts_sample <- read.csv('/home/yjseo/ulms_2021/ulms_list/data/TCGA_test_samples_2.csv', header=TRUE, stringsAsFactors=FALSE)

tr_uterus_data <- tr_sample %>% inner_join(uterus_data_T, by='sample')
ts_uterus_data <- ts_sample %>% inner_join(uterus_data_T, by='sample')
tr_uterus_data$state <- as.factor(tr_uterus_data$state)
ts_uterus_data$state <- as.factor(ts_uterus_data$state)
summary(tr_uterus_data$state)
summary(ts_uterus_data$state)


# 1. feature selection - high expressed gene 30 %
avg_exp <- tr_uterus_data[-c(1:2)] %>% apply(2,mean) %>% as.data.frame() %>% rownames_to_column('gene')
colnames(avg_exp)[2] <- 'avg'

avg_exp <- avg_exp %>% arrange(desc(avg))
high_rank_gene <- avg_exp[1:as.integer(nrow(avg_exp)*0.3),'gene']   # 6795 genes

high_rank_uterus_data <- cbind(tr_uterus_data[1:2], tr_uterus_data[high_rank_gene])
high_rank_test_data <- cbind(ts_uterus_data[1:2], ts_uterus_data[high_rank_gene])

# 1.2 feature selection 
theragen <- readRDS('/nfs-data/ULMS/ntp/data/theragen_log2_scale_clinical_data.rds')

# GAPDH scale
'ENSG00000111640' %in% colnames(theragen)
gapdh_value <- theragen['ENSG00000111640']$ENSG00000111640

head(theragen['ENSG00000111640']) # scale 전
scaled_theragen <- cbind(sample=theragen[,c(1)],
                         theragen[,-c(1)] %>% apply(2, function(x){x - gapdh_value}) %>% as.data.frame())
head(scaled_theragen['ENSG00000111640']) # scale 후
summary(scaled_theragen[1:5, 1:5])

# train set, 1st test set에서 LM 필터링
theragen_label <- read.csv('/nfs-data/ULMS/ntp/data/theragen_sample_with_label.csv', header=TRUE, stringsAsFactors=FALSE)
scaled_theragen_1st_ctl <- theragen_label[1:18,] %>% filter(state == 0)
scaled_theragen_1st <- scaled_theragen[scaled_theragen$sample %in% scaled_theragen_1st_ctl$sample,]
high_rank_uterus_data_ctl <- high_rank_uterus_data %>% filter(state == 0)

sub_ensg <- c()

for (ensg in colnames(high_rank_uterus_data_ctl)[-c(1:2)]) {
  
  if (ensg=='ENSG00000111640') {
    next
  }
  
  var_result <- var.test(high_rank_uterus_data_ctl[,ensg], scaled_theragen_1st[,ensg])
  
  if (var_result$p.value > 0.05) {
    t_result <- t.test(high_rank_uterus_data_ctl[,ensg], scaled_theragen_1st[,ensg], var.equal=TRUE)
    
    if (t_result$p.value > 0.05) {
      sub_ensg <- c(sub_ensg, ensg)
      
    }
  }
}


# 인종차 제거 유전자 = sub_ensg # 1314 genes

# 2. sample 평균을 0으로 맞춰주기
tr_sample_mean <- high_rank_uterus_data[,sub_ensg] %>% apply(1, mean)
ts_sample_mean <- high_rank_test_data[,sub_ensg] %>% apply(1, mean)
theragen_sample_mean <- scaled_theragen[,sub_ensg] %>% apply(1, mean)


tr_adjust <- cbind(high_rank_uterus_data[,c(1,2)],
                   high_rank_uterus_data[,sub_ensg] %>% apply(2, function(x){x - tr_sample_mean}) %>% as.data.frame())
ts_adjust <- cbind(high_rank_test_data[,c(1,2)],
                   high_rank_test_data[,sub_ensg] %>% apply(2, function(x){x - ts_sample_mean}) %>% as.data.frame())
theragen_label <- read.csv('/nfs-data/ULMS/ntp/data/theragen_sample_with_label.csv', header=TRUE, stringsAsFactors=FALSE)
theragen_adjust <- inner_join(theragen_label, cbind(sample = scaled_theragen[,'sample'], scaled_theragen[,sub_ensg] %>% apply(2, function(x){x - theragen_sample_mean}) %>% as.data.frame()))


write.csv(tr_adjust, '/home/yjseo/ulms_2021/ulms_list/data/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv', row.names=FALSE)
write.csv(ts_adjust, '/home/yjseo/ulms_2021/ulms_list/data/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv', row.names=FALSE)
write.csv(theragen_adjust, '/home/yjseo/ulms_2021/ulms_list/data/GAPDH_scale_trainNDFGR_zerosum_theragen.csv', row.names=FALSE)

write.csv(tr_adjust, '/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv', row.names=FALSE)
write.csv(ts_adjust, '/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv', row.names=FALSE)
write.csv(theragen_adjust, '/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_theragen.csv', row.names=FALSE)

# 3. MSE 계산
normal_mse <- tr_adjust %>% filter(state == '0') %>% dplyr::select(-sample,-state) %>% apply(2, function(x){mean(x^2)}) %>% as.data.frame()
cancer_mse <- tr_adjust %>% filter(state == '1') %>% dplyr::select(-sample,-state) %>% apply(2, function(x){mean(x^2)}) %>% as.data.frame()
#normal_mse <- tr_adjust %>% filter(state == '0') %>% select(-sample,-state) %>% apply(2, function(x){mean(x^2)}) %>% as.data.frame()
#cancer_mse <- tr_adjust %>% filter(state == '1') %>% select(-sample,-state) %>% apply(2, function(x){mean(x^2)}) %>% as.data.frame()
theta <- (cancer_mse / normal_mse) %>% rownames_to_column('gene_name') %>% arrange(desc(.))
colnames(theta)[2] <- 'mse_ratio'
 write.csv(theta, '/home/yjseo/ulms_2021/ulms_list/data/GAPDH_scale_trainNDFGR_zerosum_MSE_ratio.csv')
 write.csv(theta, '/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_MSE_ratio.csv')

# gene list table
mse_ratio <- cancer_mse / normal_mse
mse_data = cbind('Caner MSE' = cancer_mse, normal_mse) %>% cbind(mse_ratio) %>% rownames_to_column('id')
colnames(mse_data)[-1] <- c('Cancer MSE', 'Normal MSE', 'MSE Ratio')
mse_data_symbol <- inner_join(gene_data[1:2],mse_data )%>% arrange(desc(`MSE Ratio`))
write.csv(mse_data_symbol, '/home/yjseo/ulms_2021/ulms_list/data/GAPDH_scale_trainNDFGR_zerosum_MSE_cancer_normal_ratio_symbol.csv')
write.csv(mse_data_symbol, '/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_MSE_cancer_normal_ratio_symbol.csv')

plot(theta$.[1:100], xlab='gene', ylab='MSE ratio') # plot을 봤을 때 saturate
gene_20 <- gene_data[which(gene_data$id %in% theta$gene_name[1:20]),c(1,2)] %>% tibble::column_to_rownames('id')
gene_20[theta$gene_name[1:20],]




############ Visualization #########################
######### 데이터 준비 ##############
theta <- read.csv('/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_MSE_cancer_normal_ratio_symbol.csv', stringsAsFactors = FALSE, header = TRUE)
tr_data <- read.csv('/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv', stringsAsFactors = FALSE, header = TRUE)
ts_data <- read.csv('/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv', stringsAsFactors = FALSE, header = TRUE)
theragen_data <- read.csv( '/nfs-data/ULMS/Diane/data/GAPDH_scale_trainNDFGR_zerosum_theragen.csv', stringsAsFactors = FALSE, header = TRUE)

gene_data <- read_delim('/nfs-data/ULMS/data/gencode.v23.annotation.gene.txt', '\t')
gene_data$id <- substr(gene_data$id,1,15)


deg <- theta$id[1:20]

data.train <- tr_data[, c('sample', 'state', deg)]
data.train$state <- as.factor(data.train$state)
data.test <- ts_data[, c('sample', 'state', deg)]
data.test$state <- as.factor(data.test$state)
theragen_extract1 <- theragen_data[1:18, c('sample', 'state', deg)]
theragen_extract2 <- theragen_data[-c(1:18), c('sample', 'state', deg)]
theragen_extract1$state <- as.factor(theragen_extract1$state)
theragen_extract2$state <- as.factor(theragen_extract2$state)


#### combine data
gene_name <- gene_data[which(gene_data$id %in% deg),c(1,2)] %>% tibble::column_to_rownames('id')
colnames(data.train)[-c(1,2)] <- gene_name[colnames(data.train)[-c(1,2)],]
colnames(data.test)[-c(1,2)] <- gene_name[colnames(data.test)[-c(1,2)],]
colnames(theragen_extract1)[-c(1,2)] <- gene_name[colnames(theragen_extract1)[-c(1,2)],]
colnames(theragen_extract2)[-c(1,2)] <- gene_name[colnames(theragen_extract2)[-c(1,2)],]

data.train$data_set <- "Training"
data.test$data_set <- "Validation"
theragen_extract1$data_set <- "Test - 1st"
theragen_extract2$data_set <- "Test - 2nd"

com_data_public <- rbind(data.train, data.test)
levels(com_data_public$state) <- c("Normal","Leiomyosarcoma")

com_data_test <- rbind(theragen_extract1, theragen_extract2)
levels(com_data_test$state) <- c("Leiomyoma","Leiomyosarcoma")

# if public data, 'com_data_public' and if test data, 'com_data_test'
com_data <- rbind(com_data_public,com_data_test)
colnames(com_data)[2] <- "Label"
com_data$data_set <- as.factor(com_data$data_set)
com_data$data_set <- factor(com_data$data_set, levels = c("Training", "Validation", "Test - 1st", "Test - 2nd"))
com_data$Label <- factor(com_data$Label, levels = c("Normal", "Leiomyoma", "Leiomyosarcoma"))

###### heatmap ######
set.seed(15)
ha <- HeatmapAnnotation(Label = com_data$Label, Dataset = com_data$data_set,
                        annotation_legend_param = list(Label = list(title = "Label", title_gp = gpar(fontsize = 11, fontface = "bold"),
                                                                    labels_gp = gpar(fontsize = 11)),
                                                       Dataset = list(title = "Dataset", title_gp = gpar(fontsize = 11, fontface = "bold"),
                                                                      labels_gp = gpar(fontsize = 11))),
                        col = list(Label = c("Normal" = "#2A088A", "Leiomyoma" = "#08ABBC", "Leiomyosarcoma" = "#A91133"),
                                   Dataset = c("Training" = "#E8C8FE", "Validation" = "#F778D5", "Test - 1st" = "#A3C6F2", "Test - 2nd" = "#79B1F4")))
gene <- t(as.matrix(com_data[-c(1,2,dim(com_data)[2])]))
colnames(gene) <- NULL

# png("/nfs-data/ULMS/Diane/plot/fig3_heatmap.png", width=800, height=600)
Heatmap(gene,
        top_annotation = ha,
        # name = "GAPDH scale",
        heatmap_legend_param = list(title = " ", title_gp = gpar(fontsize = 11, fontface = "bold"),
                                    labels_gp = gpar(fontsize = 11)),
        cluster_rows = FALSE
        # column_title = "Heatmap : uterus normal & ulms (GAPDH-scale / ntp / gene 30)"
)
# dev.off()

# gene graph 그리기
com_data$data_set <- as.factor(com_data$data_set)
com_data$Label <- as.vector(com_data$Label)
com_data$type <- ifelse(com_data$data_set == "Training" | com_data$data_set == "Validation",
                        com_data$Label,
                        ifelse(com_data$Label == 'Leiomyoma','theragen_normal','theragen_cancer'))
data <- com_data
gene_name <- gene_data[which(gene_data$id %in% colnames(data)),c(1,2)] %>% tibble::column_to_rownames('id')


gene_data_gather1 <- data %>% filter(type == 'Leiomyosarcoma')%>% tidyr::gather(gene_name,gene_value,-sample,-Label, -data_set,-type)
gene_data_gather2 <- data %>% filter(type == 'Normal')%>% tidyr::gather(gene_name,gene_value,-sample,-Label, -data_set,-type)
gene_data_gather3 <- data %>% filter(type == 'theragen_cancer')%>% tidyr::gather(gene_name,gene_value,-sample,-Label, -data_set,-type)
gene_data_gather4 <- data %>% filter(type == 'theragen_normal')%>% tidyr::gather(gene_name,gene_value,-sample,-Label, -data_set,-type)


# train, validation, theragen 1st, 2nd 나눠서 그리기
# training
gene_data_gather1_train <- gene_data_gather1 %>% filter(data_set =='Training')
gene_data_gather2_train <- gene_data_gather2 %>% filter(data_set =='Training')

line_train <- ggplot(gene_data_gather2_train)+
  geom_line( mapping = aes(x = gene_name,y = gene_value,group = sample, colour = Label), lwd=0.4) +
  geom_line(gene_data_gather1_train, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.4) +
  
  ggtitle("Training") +
  scale_colour_manual(values=c(Normal="#00BFC4", Leiomyosarcoma="#F8766D")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5 )) +
  lims( y = c(-15,5)) +
  theme(axis.title = element_text( size = 14),
        plot.title = element_text( size = 20, hjust = 0.5),
        axis.text.x = element_text(size=14, angle = 90, hjust = 1, vjust = 0.5 )) +
  scale_x_discrete(limits=colnames(data)[-c(1,2,23,24)]) +
  theme(legend.title = element_blank(),
        legend.position = c(0.8, 0.1)) +
  theme(legend.text = element_text( size = 13),
        panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major.y=element_line(colour="grey90",linetype="solid")) +
  #theme(legend.position = "top") +
  guides(color = guide_legend(reverse = TRUE)) +
  labs( x = " ", y="GAPDH scale & zerosum normalized value")
#dev.off()

# validation
gene_data_gather1_test <- gene_data_gather1 %>% filter(data_set =='Validation')
gene_data_gather2_test <- gene_data_gather2 %>% filter(data_set =='Validation')

line_vali <- ggplot(gene_data_gather2_test)+
  geom_line( mapping = aes(x = gene_name,y = gene_value,group = sample, colour = Label), lwd=0.4) +
  geom_line(gene_data_gather1_test, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.4) +
  
  ggtitle("Validation") +
  scale_colour_manual(values=c(Normal="#00BFC4", Leiomyosarcoma="#F8766D")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5 )) +
  lims( y = c(-15,5)) +
  theme(axis.title = element_text( size = 14),
        plot.title = element_text( size = 20, hjust = 0.5),
        axis.text.x = element_text(size=14, angle = 90, hjust = 1, vjust = 0.5 )) +
  scale_x_discrete(limits=colnames(data)[-c(1,2,23,24)]) +
  theme(legend.title = element_blank(),
        legend.position = c(0.8, 0.1)) +
  theme(legend.text = element_text( size = 13),
        panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major.y=element_line(colour="grey90",linetype="solid")) +
  guides(color = guide_legend(reverse = TRUE)) +
  #theme(legend.position = "top") +
  labs( x = " ", y="GAPDH scale & zerosum normalized value")
#dev.off()

# 1st
gene_data_gather3_1st <- gene_data_gather3 %>% filter(data_set =="Test - 1st")
gene_data_gather4_1st <- gene_data_gather4 %>% filter(data_set =="Test - 1st")

gene_data_gather3_1st_wrong <- gene_data_gather3_1st[which(gene_data_gather3_1st$sample %in% c('EXP:B16-01102:FPKM','EXP:B16-01520:FPKM','EXP:B12-01095:FPKM','EXP:B12-01062:FPKM')),] 
gene_data_gather4_1st_wrong <- gene_data_gather4_1st[which(gene_data_gather4_1st$sample %in% c('EXP:B16-01102:FPKM','EXP:B16-01520:FPKM','EXP:B12-01095:FPKM','EXP:B12-01062:FPKM')),] 

gene_data_gather3_1st_wrong$Label <- 'Prediction: Leiomyoma / Actual: Leiomyosarcoma'
gene_data_gather4_1st_wrong$Label <- 'Prediction: Leiomyosarcoma / Actual: Leiomyoma'

line_1st_test <- ggplot(gene_data_gather3_1st)+
  geom_line( mapping = aes(x = gene_name,y = gene_value,group = sample, colour = Label), lwd=0.4) +
  geom_line(gene_data_gather4_1st, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.4) +
  # geom_line(gene_data_gather3_1st_wrong, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.6) +
  # geom_line(gene_data_gather4_1st_wrong, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.6) +
  
  ggtitle("Test - 1st") +
  scale_colour_manual(values=c(Leiomyoma="#00BFC4", Leiomyosarcoma="#F8766D", `Prediction: Leiomyoma / Actual: Leiomyosarcoma` = 'red', `Prediction: Leiomyosarcoma / Actual: Leiomyoma` = 'blue')) +
  theme() +
  lims( y = c(-15,5)) +
  theme(axis.title = element_text( size = 14),
        plot.title = element_text( size = 20, hjust = 0.5),
        axis.text.x = element_text(size=14, angle = 90, hjust = 1, vjust = 0.5 )) +
  scale_x_discrete(limits=colnames(data)[-c(1,2,23,24)]) +
  theme(legend.title = element_blank(),
        legend.position = c(0.8, 0.1)) +
  theme(legend.text = element_text( size = 13),
        panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major.y=element_line(colour="grey90",linetype="solid")) +
  #theme(legend.position = "top") +
  labs( x = " ", y="GAPDH scale & zerosum normalized value")
#dev.off()

# 2nd
gene_data_gather3_2nd <- gene_data_gather3 %>% filter(data_set =="Test - 2nd")
gene_data_gather4_2nd <- gene_data_gather4 %>% filter(data_set =="Test - 2nd")

gene_data_gather3_2nd_wrong <- gene_data_gather3_2nd[which(gene_data_gather3_2nd$sample %in% c('EXP:OC17-227T:FPKM','EXP:130110004:FPKM')),] 
gene_data_gather4_2nd_wrong <- gene_data_gather4_2nd[which(gene_data_gather4_2nd$sample %in% c('EXP:OC17-227T:FPKM','EXP:130110004:FPKM')),] 

gene_data_gather3_2nd_wrong$Label <- 'Prediction: Leiomyoma / Actual: Leiomyosarcoma'
gene_data_gather4_2nd_wrong$Label <- 'Prediction: Leiomyosarcoma / Actual: Leiomyoma'

line_2nd_test<- ggplot(gene_data_gather3_2nd)+
  geom_line( mapping = aes(x = gene_name,y = gene_value,group = sample, colour = Label), lwd=0.4) +
  geom_line(gene_data_gather4_2nd, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.4) +
  # geom_line(gene_data_gather3_2nd_wrong, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.6) +
  # geom_line(gene_data_gather4_2nd_wrong, mapping = aes(x = gene_name,y = gene_value, group = sample, colour = Label), lwd=0.6) +
  
  ggtitle("Test 2nd") +
  scale_colour_manual(values=c(Leiomyoma="#00BFC4", Leiomyosarcoma="#F8766D", `Prediction: Leiomyoma / Actual: Leiomyosarcoma` = 'red', `Prediction: Leiomyosarcoma / Actual: Leiomyoma` = 'blue')) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5 )) +
  lims( y = c(-15,5)) +
  theme(axis.title = element_text( size = 14),
        plot.title = element_text( size = 20, hjust = 0.5),
        axis.text.x = element_text(size=14, angle = 90, hjust = 1, vjust = 0.5 )) +
  scale_x_discrete(limits=colnames(data)[-c(1,2,23,24)]) +
  theme(legend.title = element_blank(),
        legend.position = c(0.8, 0.1)) +
  theme(legend.text = element_text( size = 13),
        panel.background = element_rect(fill = 'white', colour = 'gray'),
        panel.grid.major.y=element_line(colour="grey90",linetype="solid")) +
  #theme(legend.position = "top") +
  labs( x = " ", y="GAPDH scale & zerosum normalized value")
# dev.off()

library(gridExtra)
png("/nfs-data/ULMS/Diane/plot/fig2_line_graph.png", width = 900, height = 900)
grid.arrange(line_train, line_vali, line_1st_test, line_2nd_test, nrow=2, ncol=2)
dev.off()

our_deg <- colnames(data)[-c(1,2,23,24)]
other_deg <- c("BCAN", "AAK1", "PCBP3", "MOV10L1", "TWISTNB", "TMSB15A", "SMAD1", "ANXA1", "FOS", "SLFN11" )
which(our_deg %in% other_deg)
which(other_deg %in% our_deg)
which(gene_data$gene %in% other_deg)
