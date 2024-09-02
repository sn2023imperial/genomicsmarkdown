import numpy as np
import pandas as pd
import grelu.data.preprocess as preprocess
import grelu.data.dataset
import grelu.lightning
import grelu.visualize
import pickle
from sklearn.utils import resample

def main():
    # Model constants
 
    PRED_RES = 512
    val_chroms = "chr3"
    test_chroms = "chr2"
    genome = "hg38"
    bw_file =  'raw_data/ENCFF601VTB.bigWig' #p-value #'ENCFF359FNY.bigWig' #fold change



    # Load our DNA bins filled with our target values
    dna_bins = pd.read_csv('raw_data/512bpResolution_p_values_ALLCHROMS.csv.gz')  # Intervals and coverage from chromosome's 1-5 with a 512bp resolution (summed)
   
    # Set a threshold to filter our data
    threshold = 2
    thresholdarc = np.arcsinh(threshold) # The threshold has to undergo an arcsinh transformation as well

   
    # Apply the threshold to all chromosomes except chrom2 and chrom3
    filt_dna_bins = dna_bins[(dna_bins['cov'] > thresholdarc) & 
                            ~(dna_bins['chrom'].isin([test_chroms, val_chroms]))]

  
    # Output the shape to verify
    print(f"{filt_dna_bins.shape[0]} training/validation/test positions.")
    
    
    
    # Keep only the 'chrom', 'start', and 'end' columns (gRelu naming standards)
    peaks = filt_dna_bins[['chrom', 'start', 'end']]
    

    print(peaks) # This gives us intervals from thresholded peaks e.g. locations we want to explore

    # Remove peaks near blacklist region
    peaks = preprocess.filter_blacklist(
            peaks,
            genome=genome,
            window=50 # within 50 bp of a blacklist region
        )
    
    # Add negative regions to our datasets (not on chrom2, 3)
    negatives = preprocess.get_gc_matched_intervals(
        peaks,
        binwidth=0.02, # resolution of measuring GC content
        genome=genome,
        chroms=['chr1', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
                'chr20', 'chr21', 'chr22'], 
        gc_bw_file='gc_hg38_2048.bw',
        blacklist=genome, # negative regions overlapping the blacklist will be dropped
        seed=0,
        )
    negatives.head(3)

    regions = pd.concat([peaks, negatives]).reset_index(drop=True)
    len(regions)
    
    print(regions)
    

    # Sampling our validation and test sets to 1500
    valid_size = 1500
    test_size = 1500
    
     # 1. Filter the data by chromosome
    chr2_data = regions[regions['chrom'] == test_chroms]
    chr3_data = regions[regions['chrom'] == val_chroms]
    train_data = regions[~regions['chrom'].isin([test_chroms, val_chroms])]

    # 2. Downsample chr2 and chr3 data if necessary
    if len(chr2_data) > test_size:
        chr2_data = resample(chr2_data, n_samples=test_size, random_state=1)

    if len(chr3_data) > valid_size:
        chr3_data = resample(chr3_data, n_samples=valid_size, random_state=1)


    # 4. Combine the final training, validation, and test sets
    train = train_data
    val = chr3_data
    test = chr2_data

   


    


    # Creating gRelu datasets - gRelu adds the coverage values from a bigwig, summing + arcsinh transfromation
    train_ds = grelu.data.dataset.BigWigSeqDataset(
        intervals = train,
        bw_files=[bw_file],
        label_len=PRED_RES,
        label_aggfunc="sum",
        seed=0,
        genome=genome,
        label_transform_func=np.arcsinh
    )



    val_ds = grelu.data.dataset.BigWigSeqDataset(
        intervals = val,
        bw_files=[bw_file],
        label_len=PRED_RES,
        label_aggfunc="sum", 
        genome=genome,
        label_transform_func=np.arcsinh
    )
    

    test_ds = grelu.data.dataset.BigWigSeqDataset(
        intervals = test,
        bw_files=[bw_file],
        label_len=PRED_RES,
        label_aggfunc="sum",
        genome=genome,
        label_transform_func=np.arcsinh
    )

    
    len(train_ds), len(val_ds), len(test_ds)




    # Save data as pkl files
    
    with open('./datasets/unsplit_allchroms/train_ds_compare.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
        
    
    with open('./datasets/unsplit_allchroms/val_ds_compare.pkl', 'wb') as f:
        pickle.dump(val_ds, f)
    
    
    with open('./datasets/unsplit_allchroms/test_ds_compare.pkl', 'wb') as f:
        pickle.dump(test_ds, f)
        
    

if __name__ == '__main__':
    main()