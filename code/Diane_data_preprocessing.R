##### Library import #####
library(readr)
library(dplyr)
library(tibble)


##### Data loading #####
# Public data
data <- read_delim("../data/TcgaTargetGtex_rsem_gene_fpkm", "\t", escape_double = FALSE, trim_ws = TRUE)
phenotype <- read.csv('../data/TcgaTargetGTEX_phenotype.csv',header = TRUE)
ULMS_list <- read.csv("../data/TcgaTargetGtex_ulms_list.csv",header = TRUE)
Uterus_list <- read.csv("../data/TcgaTargetGTEX_Uterus_list.csv",header = TRUE)
gene_data <- read_delim('../data/gencode.v23.annotation.gene.txt', '\t')
inter_gene <- read.csv('../data/TCGA_theragen_inter_gene.csv', header=TRUE, stringsAsFactors=FALSE)

# Clinical sample data
clinical <- readRDS('../data/theragen_log2_scale_clinical_data.rds')
clinical_label <- read.csv('../data/theragen_sample_with_label.csv', header=TRUE, stringsAsFactors=FALSE)
test_sample_1st <- clinical_label[1:18,]
test_sample_2nd <- clinical_label[19:35,]


##### Prepare intersected gene data with uterus sample only #####
data$sample <- substr(data$sample,1,15)
gene_data$id <- substr(gene_data$id,1,15)

inter_gene_data <- data[data$sample %in% inter_gene$sample,]  # Extract intersected genes between Public data and clinical samples
rownames(inter_gene_data) <- NULL
uterus_data <- cbind(sample=inter_gene_data[1], inter_gene_data[,colnames(inter_gene_data) %in% Uterus_list$sample])


##### Within-sample standardization using GAPDH gene #####
# Public data
uterus_data_T <- uterus_data %>% column_to_rownames('sample') %>% t() %>% as.data.frame() %>% rownames_to_column('sample')
gapdh_value <- uterus_data_T$ENSG00000111640

uterus_data_T_scaled <- cbind(sample=uterus_data_T[,c(1)],
                              uterus_data_T[,-c(1)] %>% apply(2, function(x){x - gapdh_value}) %>% as.data.frame())

# Clinical sample data
clinical_gapdh_value <- clinical$ENSG00000111640
scaled_clinical <- cbind(sample=clinical[,c(1)],
                         clinical[,-c(1)] %>% apply(2, function(x){x - clinical_gapdh_value}) %>% as.data.frame())


##### Data splitting #####
tr_sample <- read.csv('../data/TCGA_train_samples_2.csv', header=TRUE, stringsAsFactors=FALSE)
ts_sample <- read.csv('../data/TCGA_test_samples_2.csv', header=TRUE, stringsAsFactors=FALSE)

tr_uterus_data <- tr_sample %>% inner_join(uterus_data_T_scaled, by='sample')
ts_uterus_data <- ts_sample %>% inner_join(uterus_data_T_scaled, by='sample')
tr_uterus_data$state <- as.factor(tr_uterus_data$state)
ts_uterus_data$state <- as.factor(ts_uterus_data$state)


##### Feature selection ##### 
# 1. High expressed gene 30 %
avg_exp <- tr_uterus_data[-c(1:2)] %>% apply(2,mean) %>% as.data.frame() %>% rownames_to_column('gene')
colnames(avg_exp)[2] <- 'avg'

avg_exp <- avg_exp %>% arrange(desc(avg))
high_rank_gene <- avg_exp[1:as.integer(nrow(avg_exp)*0.3),'gene']

high_rank_uterus_data <- cbind(tr_uterus_data[1:2], tr_uterus_data[high_rank_gene])
high_rank_test_data <- cbind(ts_uterus_data[1:2], ts_uterus_data[high_rank_gene])


# 2. Select NDEG (Not Differentially Expressed Genes between Normal and Leiomyoma samples)
scaled_clinical_1st <- scaled_clinical[scaled_clinical$sample %in% test_sample_1st$sample,]

tr_normal_sample <- (tr_sample %>% filter(state==0))$sample
leiomyoma_sample <- (test_sample_1st %>% filter(state==0))$sample # Used 1st independent test set only

sub_ensg <- c()
for (ensg in high_rank_gene) {
  if (ensg=='ENSG00000111640') {
    next
  }
  
  var_result <- var.test(high_rank_uterus_data[high_rank_uterus_data$sample %in% tr_normal_sample, ensg], 
                         scaled_clinical_1st[scaled_clinical_1st$sample %in% leiomyoma_sample,ensg])
  
  if (var_result$p.value > 0.05) {
    t_result <- t.test(high_rank_uterus_data[high_rank_uterus_data$sample %in% tr_normal_sample,ensg], 
                       scaled_clinical_1st[scaled_clinical_1st$sample %in% leiomyoma_sample,ensg], var.equal=TRUE)
    
    if (t_result$p.value > 0.05) {
      sub_ensg <- c(sub_ensg, ensg)
      
    }
  }
}


# 3. Zero-sum Transformation
tr_sample_mean <- high_rank_uterus_data[,sub_ensg] %>% apply(1, mean)
ts_sample_mean <- high_rank_test_data[,sub_ensg] %>% apply(1, mean)
clinical_sample_mean <- scaled_clinical[,sub_ensg] %>% apply(1, mean)

tr_adjust <- cbind(high_rank_uterus_data[,c(1,2)],
                   high_rank_uterus_data[,sub_ensg] %>% apply(2, function(x){x - tr_sample_mean}) %>% as.data.frame())
ts_adjust <- cbind(high_rank_test_data[,c(1,2)],
                   high_rank_test_data[,sub_ensg] %>% apply(2, function(x){x - ts_sample_mean}) %>% as.data.frame())

clinical_adjust <- inner_join(clinical_label, cbind(sample = scaled_clinical[,'sample'], 
                                                    scaled_clinical[,sub_ensg] %>% apply(2, function(x){x - clinical_sample_mean}) %>% as.data.frame()))

# write.csv(tr_adjust, '../data_revised/GAPDH_scale_trainNDFGR_zerosum_TCGAtrain.csv', row.names=FALSE)
# write.csv(ts_adjust, '../data_revised/GAPDH_scale_trainNDFGR_zerosum_TCGAtest.csv', row.names=FALSE)
# write.csv(clinical_adjust, '../data_revised/GAPDH_scale_trainNDFGR_zerosum_theragen.csv', row.names=FALSE)


# 4. MSE ratio calculation
normal_mse <- tr_adjust %>% filter(state == '0') %>% dplyr::select(-sample,-state) %>% apply(2, function(x){mean(x^2)}) %>% as.data.frame()
cancer_mse <- tr_adjust %>% filter(state == '1') %>% dplyr::select(-sample,-state) %>% apply(2, function(x){mean(x^2)}) %>% as.data.frame()

theta <- (cancer_mse / normal_mse) %>% rownames_to_column('gene_name') %>% arrange(desc(.))
colnames(theta)[2] <- 'mse_ratio'

# write.csv(theta, '../data_revised/GAPDH_scale_trainNDFGR_zerosum_MSE_ratio.csv')


