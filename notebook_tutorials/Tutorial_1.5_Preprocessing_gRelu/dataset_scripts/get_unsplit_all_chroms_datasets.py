import numpy as np
import pandas as pd
import grelu.data.preprocess as preprocess
import grelu.data.dataset
import grelu.lightning
import grelu.visualize
import pickle

def main():
    # Model constants
 
    PRED_RES = 512
    #val_chroms =  validating on all chroms (separate set)
    #test_chroms =  testing on all chroms (separate set)
    genome = "hg38"
    bw_file =  'raw_data/ENCFF601VTB.bigWig' #p-value #'ENCFF359FNY.bigWig' #fold change



    # Load our DNA bins filled with our target values
    dna_bins = pd.read_csv('raw_data/512bpResolution_p_values_ALLCHROMS.csv.gz')  # Intervals and coverage from chromosome's 1-5 with a 512bp resolution (summed)
    
    # Separating out real distributions of all chroms for val and test set
    val_test = dna_bins.sample(frac=1, random_state=2).reset_index(drop=True)
    # Val/Test set split (we will remove from our training set later)
    v = val_test.iloc[:1500]
    t = val_test.iloc[1500:3000]
    validation_set = v[['chrom', 'start', 'end']]
    test_set = t[['chrom', 'start', 'end']]
    
    # Set a threshold to filter our data
    threshold = 2
    thresholdarc = np.arcsinh(threshold) # The threshold has to undergo an arcsinh transformation as well


    # Thresholding
    filt_dna_bins = dna_bins.loc[dna_bins['cov']>thresholdarc].reset_index(drop=True) # Apply the threshold to all chromosomes
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
    
    # Remove peaks near blacklist region
    validation_set = preprocess.filter_blacklist(
            validation_set,
            genome=genome,
            window=50 # within 50 bp of a blacklist region
        )
    
    # Remove peaks near blacklist region
    test_set = preprocess.filter_blacklist(
            test_set,
            genome=genome,
            window=50 # within 50 bp of a blacklist region
        )
    
    # Add negative regions to our dataset (all chroms)
    negatives = preprocess.get_gc_matched_intervals(
        peaks,
        binwidth=0.02, # resolution of measuring GC content
        genome=genome,
        chroms=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
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
    

    
    # Dropping 3000 val/test samples from regions
    combined_set = pd.concat([validation_set, test_set])

    # Merge regions with combined_set to find the matching rows
    merged = regions.merge(combined_set, on=['chrom', 'start', 'end'], how='inner')

    # Drop the matching rows from regions
    regions_filtered = regions.drop(merged.index)
    
    
    # Dropping 3000 negative samples

    # Step 2: Randomly sample 3,000 rows from this subset
    rows_to_drop = negatives.sample(n=3000, random_state=42)

    # Step 3: Merge to find the matching rows
    regions_filtered = regions_filtered.merge(rows_to_drop[['chrom', 'start', 'end']],
                                            on=['chrom', 'start', 'end'],
                                            how='left', 
                                            indicator=True)

    # Step 4: Drop the rows where the indicator column is 'both' (indicating a match)
    regions_filtered = regions_filtered[regions_filtered['_merge'] == 'left_only']

    # Drop the _merge column and reset index
    regions_filtered = regions_filtered.drop(columns=['_merge']).reset_index(drop=True)


    # Reducing to match comparison model training size = 121375 rows
    regions_filtered = regions_filtered.sample(n=121375, random_state=42).reset_index(drop=True)
    # Display the resulting DataFrame
    print(regions_filtered)


    # Creating gRelu datasets - gRelu adds the coverage values from a bigwig, summing + arcsinh transfromation
    # In this case, regions_filtered is our training set
    train_ds = grelu.data.dataset.BigWigSeqDataset(
        intervals = regions_filtered,
        bw_files=[bw_file],
        label_len=PRED_RES,
        label_aggfunc="sum",
        seed=0,
        genome=genome,
        label_transform_func=np.arcsinh
    )



    val_ds = grelu.data.dataset.BigWigSeqDataset(
        intervals = validation_set,
        bw_files=[bw_file],
        label_len=PRED_RES,
        label_aggfunc="sum", 
        genome=genome,
        label_transform_func=np.arcsinh
    )
    

    test_ds = grelu.data.dataset.BigWigSeqDataset(
        intervals = test_set,
        bw_files=[bw_file],
        label_len=PRED_RES,
        label_aggfunc="sum",
        genome=genome,
        label_transform_func=np.arcsinh
    )

    
    len(train_ds), len(val_ds), len(test_ds)




    # Save data as pkl files
    
    with open('./datasets/unsplit_allchroms/train_ds_unsplit.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
        
    
    with open('./datasets/unsplit_allchroms/val_ds_unsplit.pkl', 'wb') as f:
        pickle.dump(val_ds, f)
    
    
    with open('./datasets/unsplit_allchroms/test_ds_unsplit.pkl', 'wb') as f:
        pickle.dump(test_ds, f)
        
    

if __name__ == '__main__':
    main()