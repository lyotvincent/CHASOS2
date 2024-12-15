# Explain the data file in this folder
* ChIA-PET
* * CTCF
* * * bedpe
* * * * **.e500.clusters.cis.BE3*
* * * bedpe_filtered
* * * * MA1929.1
* * * * * p-value_1e-4
* * * * * * **.filtered.e500.clusters.cis.BE3*
* * * package
* * * * **.bedpe.gz*
* * * *loop_record.csv*
* FIMO_result
* * *_hg38_FIMO_result
* * *_reverse_hg38_FIMO_result
* GCF_000001405.40
* * **.fasta*
* hg38
* * *hg38.fa*  

*.filtered.e500.clusters.cis.BE3中的表示anchor或motif的位置都是设染色体第一个位置是0。

positive_loop和negative_loop里的_v2是在motif_strength_adder.py里生成的。