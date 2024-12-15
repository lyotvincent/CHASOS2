# ChIA-PET2 need bwa index
# -g|--genome:    genome index for bwa
# ! bwa don't generate .sa file while RAM=4GB. And change RAM size to 8GM solve it.
bwa index hg38.fa

# nohup [command] > nohup.out 2>&1 &
# watch -n 5 tail -n 10 nohup.out # watch 5s a time & print final 10 lines
# run ChIA-PET2 example
# ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/GM12878/read1_ENCFF409UJV.fastq.gz -r ./raw_data/GM12878/read2_ENCFF810FRZ.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o GM12878_ChIA-PET2_result -n GM12878 -m 1 -t 8
    # run environment:
    # bedtools                  2.29.2
    # bwa                       0.7.17
    # macs2                     2.1.4
    # python                    2.7.15
    # r-base                    3.6.1
    # r-ggplot2                 3.1.1
    # r-vgam                    1.1_1
    # samtools                  1.9
## GM12878
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/GM12878/read1_ENCFF409UJV.fastq.gz -r ./raw_data/GM12878/read2_ENCFF810FRZ.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o GM12878_ChIA-PET2_result -n GM12878 -m 1 -t 8 > nohup.out 2>&1 &
## K562
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/K562/read1_ENCFF915YCD.fastq.gz -r ./raw_data/K562/read2_ENCFF245TLN.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o K562_ChIA-PET2_result -n K562 -m 1 -t 8 > nohup.out 2>&1 &
## RWPE1
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/RWPE1/read1_ENCFF096ZQA.fastq.gz -r ./raw_data/RWPE1/read2_ENCFF422NCN.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o RWPE1_ChIA-PET2_result -n RWPE1 -m 1 -t 8 > nohup.out 2>&1 &
## Caco-2
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/Caco-2/read1_ENCFF544SMM.fastq.gz -r ./raw_data/Caco-2/read2_ENCFF109BXE.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o Caco-2_ChIA-PET2_result -n Caco-2 -m 1 -t 8 > nohup.out 2>&1 &
## OCI-LY7
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/OCI-LY7/read1_ENCFF593CIG.fastq.gz -r ./raw_data/OCI-LY7/read2_ENCFF261EWD.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o OCI-LY7_ChIA-PET2_result -n OCI-LY7 -m 1 -t 8 > nohup.out 2>&1 &
## PC-3
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/PC-3/read1_ENCFF131SQE.fastq.gz -r ./raw_data/PC-3/read2_ENCFF392ZNZ.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o PC-3_ChIA-PET2_result -n PC-3 -m 1 -t 8 > nohup.out 2>&1 &
## Panc1
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/Panc1/read1_ENCFF331HZQ.fastq.gz -r ./raw_data/Panc1/read2_ENCFF937UXS.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o Panc1_ChIA-PET2_result -n Panc1 -m 1 -t 8 > nohup.out 2>&1 &
## ! Patski 在MICC阶段出现error
## ! 发现Patski的cell line是小家鼠的，可能是因此失败
## ! Mus musculus strain Patski Patski
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/Patski/read1_ENCFF144WOU.fastq.gz -r ./raw_data/Patski/read2_ENCFF446BIX.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o Patski_ChIA-PET2_result -n Patski -m 1 -t 8 > nohup.out 2>&1 &
## ! HEK293T don't have ChIP-seq data
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/HEK293T/read1_ENCFF237RPN.fastq.gz -r ./raw_data/HEK293T/read2_ENCFF102XPP.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o HEK293T_ChIA-PET2_result -n HEK293T -m 1 -t 8 > nohup.out 2>&1 &
## MCF-7
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/MCF-7/read1_ENCFF592KNN.fastq.gz -r ./raw_data/MCF-7/read2_ENCFF342KQM.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o MCF-7_ChIA-PET2_result -n MCF-7 -m 1 -t 8 > nohup.out 2>&1 &
## A549
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/A549/read1_ENCFF985XQV.fastq.gz -r ./raw_data/A549/read2_ENCFF994RVV.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o A549_ChIA-PET2_result -n A549 -m 1 -t 8 > nohup.out 2>&1 &
## A673
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/A673/read1_ENCFF967ILW.fastq.gz -r ./raw_data/A673/read2_ENCFF790CGR.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o A673_ChIA-PET2_result -n A673 -m 1 -t 8 > nohup.out 2>&1 &
## IMR-90
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/IMR-90/read1_ENCFF932VWG.fastq.gz -r ./raw_data/IMR-90/read2_ENCFF908YMO.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o IMR-90_ChIA-PET2_result -n IMR-90 -m 1 -t 8 > nohup.out 2>&1 &
## AG04450
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/AG04450/read1_ENCFF581VKF.fastq.gz -r ./raw_data/AG04450/read2_ENCFF387IFJ.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o AG04450_ChIA-PET2_result -n AG04450 -m 1 -t 8 > nohup.out 2>&1 &
## BJ
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/BJ/read1_ENCFF634VYO.fastq.gz -r ./raw_data/BJ/read2_ENCFF902YQJ.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o BJ_ChIA-PET2_result -n BJ -m 1 -t 8 > nohup.out 2>&1 &
## HCT116
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/HCT116/read1_ENCFF869HLZ.fastq.gz -r ./raw_data/HCT116/read2_ENCFF933EOS.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o HCT116_ChIA-PET2_result -n HCT116 -m 1 -t 8 > nohup.out 2>&1 &
## MCF-10A
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/MCF-10A/read1_ENCFF520GGF.fastq.gz -r ./raw_data/MCF-10A/read2_ENCFF290IDP.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o MCF-10A_ChIA-PET2_result -n MCF-10A -m 1 -t 8 > nohup.out 2>&1 &
## SK-N-SH
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/SK-N-SH/read1_ENCFF084YMC.fastq.gz -r ./raw_data/SK-N-SH/read2_ENCFF440ERH.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o SK-N-SH_ChIA-PET2_result -n SK-N-SH -m 1 -t 8 > nohup.out 2>&1 &
## AG04449
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/AG04449/read1_ENCFF868WGL.fastq.gz -r ./raw_data/AG04449/read2_ENCFF430KRH.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o AG04449_ChIA-PET2_result -n AG04449 -m 1 -t 8 > nohup.out 2>&1 &
## H1
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/H1/read1_ENCFF351NOQ.fastq.gz -r ./raw_data/H1/read2_ENCFF054HUO.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o H1_ChIA-PET2_result -n H1 -m 1 -t 8 > nohup.out 2>&1 &
## WTC11
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/WTC11/read1_ENCFF256XKZ.fastq.gz -r ./raw_data/WTC11/read2_ENCFF329OGV.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o WTC11_ChIA-PET2_result -n WTC11 -m 1 -t 8 > nohup.out 2>&1 &
## GM10248
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/GM10248/read1_ENCFF803DOQ.fastq.gz -r ./raw_data/GM10248/read2_ENCFF149FYK.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o GM10248_ChIA-PET2_result -n GM10248 -m 1 -t 8 > nohup.out 2>&1 &
## PC-9
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/PC-9/read1_ENCFF781PRX.fastq.gz -r ./raw_data/PC-9/read2_ENCFF615NPK.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o PC-9_ChIA-PET2_result -n PC-9 -m 1 -t 8 > nohup.out 2>&1 &
## HepG2
nohup ChIA-PET2 -g ./hg38/hg38.fa -b ./hg38/hg38.chrom.sizes -f ./raw_data/HepG2/read1_ENCFF285PTY.fastq.gz -r ./raw_data/HepG2/read2_ENCFF042GDX.fastq.gz -A ACGCGATATCTTATCTGACT -B AGTCAGATAAGATATCGCGT -o HepG2_ChIA-PET2_result -n HepG2 -m 1 -t 8 > nohup.out 2>&1 &


## -help
# $ ChIA-PET2 -h
# usage : ChIA-PET2 -g genomeindex -b bedtoolsgenome -f fq1 -r fq2 -A linkerA -B linkerB -o OUTdir -n prefixname
# Use option -h|--help for more information

# ChIA-PET2 0.9.3   2017.11.07
# ----------------------------
# OPTIONS

#   -s|--start:     start from which step(1:8): 1:Trim Linkers; 2:Map Reads; 3:Build PETs; 4:Call Peaks; 5:Find Interactions
#                   6:Plot QC; 7:Estimate statistical confidence; 8:Phase PETs(optional), default=1
#   -g|--genome:    genome index for bwa
#   -b|--bedtoolsgenome: chromsomes size file for bedtools
#   -f|--forward:   one fastq(.gz) file
#   -r|--reverse:   the other fastq(.gz) file
#   -A|--linkerA:   one linker sequence, default=GTTGGATAAG
#   -B|--linkerB:   the other linker sequence, default=GTTGGAATGT
#   -o|--output:    output folder, default=output
#   -n|--name:      output prefix name, default=out
#   -m|--mode:      0,1,2;  0: A/B linkers; 1: bridge liker; 2: Enzyme site, default=0
#   -e|--err:       Maximum mismatches allowed in linker sequence, default=0
#   -k|--keepempty: 0,1,2; 0:No linker-empty reads; 1:keep 1 linker-empty read; 2:keep 2 linker-empty reads. default=0
#   -t|--thread:    threads to run, default=1
#   -d|--short:     short reads (0 or 1), default=0 for reads >70bp. If the read length is about 20bp, set d=1
#   -M|--macs2 parameters, default="-q 0.05"
#   -Q|--mapq:      mapq cutoff, default=30
#   -C|--cutoffPET: PET count cutoff before running MICC, default=2
#   -S|--slop:      slop length, default=100
#   -E|--extend:    extend length on both sides, default=500
#   -l|--length:    min length of reads after linker trimming. default=15
#   -P|--phased:    optional phased genotype file: 'chr1\tstart\tend\tA\tC'
#   [-h|--help]:    help
#   [-v|--version]: version
