GTEx gene-expression-by-tissue control file downloaded from :
https://gtexportal.org/home/downloads/adult-gtex/bulk_tissue_expression
ex. "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct"

gnomAD copy number values from :
https://gnomad.broadinstitute.org/downloads#v4-structural-variants --> Genome SV BED file
ex. "gnomad.v4.1.sv.sites.bed.gz"

[!] Must be compliant with standard used in biological pipeline
GENCODE annotation file downloaded from: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/
-> standard for TCGA dataset on which the project was built and tested.






---------------------------------------

L'RNA ha una baseline quantitativa continua (GTEx): Per l'RNA sottrarsi ha senso (TPM_tumor - GTEx_baseline)
perché GTEx ti fornisce un valore quantitativo medio/mediano di espressione (es. $15.4\text{ TPM}$) per ogni singolo
gene nel tessuto normale.

gnomAD fornisce frequenze, non valori di LogR: dice solo con quale frequenza una variante strutturale o CNV si presenta
nella popolazione sana (es. AF = 0.05 cioè presente nel 5% della popolazione), ma non dà un valore continuo di LogR
del paziente da poter sottrarre numericamente dal dato tumorale.

purifiedLogR è GIÀ un Delta: Nei dati di sequenziamento del DNA tumorale, è già di per sé uno scostamento (log-ratio)
rispetto allo stato normale diploide (2n). Per definizione:

--> LogR = 0 implica "Diploide normale" (2n)
--> LogR > 0 implica "Guadagno / Amplificazione"
--> LogR < 0 implica "Delezione"

Impostare somatic_purifiedLogR = 0.0 quando un gene ha una CNV frequente nella popolazione normale (AF >= 1%) compie
un'operazione nota in bioinformatica come Germline Masking (o Filtering dei polimorfismi).
Obiettivo: Evitare che varianti costituzionali/germinali benigni e molto comuni nelle persone sane (es. duplicazioni o
delezioni polimorfiche nei geni delle difensine, amilasi, GST, ecc.) vengano scambiate nel tuo studio per alterazioni
somatiche tumorali (oncogeni o tumor suppressor).

Effetto pratico: Portando il valore a 0.0, dici all'analisi: "Questo gene varia spesso anche nelle persone sane; quindi,
non considero questa variazione come un evento somatico specifico del tumore, ma lo riporto al livello neutro diploide.