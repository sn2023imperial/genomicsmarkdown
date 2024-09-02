import numpy as np
import pandas as pd
import grelu.data.preprocess as preprocess
import grelu.data.dataset
import grelu.lightning
import grelu.visualize
import pickle
from sklearn.utils import shuffle

def main():
    # Model constants
 
    PRED_RES = 512
    val_chroms = "chr3"
    test_chroms = "chr19"
    genome = "hg38"
    bw_file =  'raw_data/ENCFF601VTB.bigWig' #p-value #'ENCFF359FNY.bigWig' #fold change



    # Load our DNA bins filled with our target values
    dna_bins = pd.read_csv('512bpResolution_p_values_ALLCHROMS.csv.gz')  # Intervals and coverage from chromosome's 1-5 with a 512bp resolution (summed)


    # Keep only the 'chrom', 'start', and 'end' columns (gRelu naming standards)
    peaks = dna_bins[['chrom', 'start', 'end']]
    desired_chromosomes = ['chr1', 'chr3', 'chr4', 'chr5', 'chr19']

    # Filter the DataFrame to include only the specified chromosomes
    peaks = peaks[peaks['chrom'].isin(desired_chromosomes)]

    print(peaks) # This gives us intervals from thresholded peaks e.g. locations we want to explore

    # Remove peaks near blacklist region
    peaks = preprocess.filter_blacklist(
            peaks,
            genome=genome,
            window=50 # within 50 bp of a blacklist region
        )
    
    regions = peaks
    
    print(regions)

    # We now have intervals of peaks and non-peaks.


    
     # Using SKLearn's resample to reduce size of validation and test sets + create data splits.
    train_size = 15000
    valid_size = 1500
    test_size = 1500

   # Shuffle the entire dataset
    regions = shuffle(regions, random_state=1)

    # Randomly select samples for the training, validation, and test sets
    train = regions.sample(n=train_size, random_state=1)
    remainder = regions.drop(train.index)
    val = remainder.sample(n=valid_size, random_state=1)
    test = remainder.drop(val.index).sample(n=test_size, random_state=1)

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
    
    with open('./datasets/unsplit_test/train_ds_unsplit.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
        
    
    with open('./datasets/unsplit_test/val_ds_unsplit.pkl', 'wb') as f:
        pickle.dump(val_ds, f)
    
    
    with open('./datasets/unsplit_test/test_ds_unsplit.pkl', 'wb') as f:
        pickle.dump(test_ds, f)
        
    

if __name__ == '__main__':
    main()