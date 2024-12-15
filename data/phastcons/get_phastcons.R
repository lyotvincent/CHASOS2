# https://bioconductor.org/packages/release/data/annotation/html/phastCons100way.UCSC.hg38.html



# library("BiocManager")
# help(package="BiocManager")
# options("repos"=c(CRAN="https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
# if (!require("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("phastCons100way.UCSC.hg38")

# install.packages("/home/sjl/workspace/phastcons/phastCons100way.UCSC.hg38_3.7.1.tar.gz")
library("phastCons100way.UCSC.hg38")
# help(package="phastCons100way.UCSC.hg38")

# print("===1")
# print(phastCons100way.UCSC.hg38)
# ls("package:phastCons100way.UCSC.hg38")
print("===2")
phast <- phastCons100way.UCSC.hg38
# phast
# citation(phast)
# gscores(phast, GRanges("chr7:117592326-117592330"))
new.function <- function(chr,start,end,out) {
    print(chr)
    # gr <- GRanges(
    #   seqnames = Rle(c("chr1", "chr2", "chr1", "chr3"), c(1, 3, 2, 4)),
    #   ranges = IRanges(start=101:110, width=1),
    #   strand = Rle(strand(c("-", "+", "*", "+", "-")), c(1, 2, 2, 3, 2)))

    gr <- GRanges(
      seqnames = Rle(c(chr), c(end-start+1)),
      ranges = IRanges(start=start:end, width=1),
      strand = Rle(strand(c("+")), c(end-start+1)))

    # a = gscores(phast, GRanges("chr1:248956422:+"))
    aresult = gscores(phast, gr)
    # gscores(phast, GRanges("chr1:1170000-1170001"))
    # print(a)
    # nsites(phast)
    # gscoresTag(phast)
    # gscoresGroup(phast)
    # populations(phast)
    # defaultPopulation(phast)

    write.csv(aresult,out)

}

# need fill from chr1 to chr5
# new.function(chr="chr6",start=1,end=100000000,out="chr6+1.csv")
# new.function(chr="chr6",start=100000001,end=170805979,out="chr6+2.csv")
# new.function(chr="chr7",start=1,end=100000000,out="chr7+1.csv")
# new.function(chr="chr7",start=100000001,end=159345973,out="chr7+2.csv")
# new.function(chr="chr8",start=1,end=100000000,out="chr8+1.csv")
# new.function(chr="chr8",start=100000001,end=145138636,out="chr8+2.csv")
# new.function(chr="chr9",start=1,end=100000000,out="chr9+1.csv")
# new.function(chr="chr9",start=100000001,end=138394717,out="chr9+2.csv")
# new.function(chr="chr10",start=1,end=100000000,out="chr10+1.csv")
# new.function(chr="chr10",start=100000001,end=133797422,out="chr10+2.csv")
# new.function(chr="chr11",start=1,end=100000000,out="chr11+1.csv")
# new.function(chr="chr11",start=100000001,end=135086622,out="chr11+2.csv")
# new.function(chr="chr12",start=1,end=100000000,out="chr12+1.csv")
# new.function(chr="chr12",start=100000001,end=133275309,out="chr12+2.csv")
# new.function(chr="chr13",start=1,end=100000000,out="chr13+1.csv")
# new.function(chr="chr13",start=100000001,end=114364328,out="chr13+2.csv")
# new.function(chr="chr14",start=1,end=100000000,out="chr14+1.csv")
# new.function(chr="chr14",start=100000001,end=107043718,out="chr14+2.csv")
# new.function(chr="chr15",start=1,end=100000000,out="chr15+1.csv")
# new.function(chr="chr15",start=100000001,end=101991189,out="chr15+2.csv")
# new.function(chr="chr16",start=1,end=90338345,out="chr16+1.csv")
# new.function(chr="chr17",start=1,end=83257441,out="chr17+1.csv")
# new.function(chr="chr18",start=1,end=80373285,out="chr18+1.csv")
# new.function(chr="chr19",start=1,end=58617616,out="chr19+1.csv")
# new.function(chr="chr20",start=1,end=64444167,out="chr20+1.csv")
# new.function(chr="chr21",start=1,end=46709983,out="chr21+1.csv")
# new.function(chr="chr22",start=1,end=50818468,out="chr22+1.csv")
new.function(chr="chrX",start=1,end=100000000,out="chrX+1.csv")
new.function(chr="chrX",start=100000001,end=156040895,out="chrX+2.csv")
new.function(chr="chrY",start=1,end=57227415,out="chrY+1.csv")


