import numpy as np
import pandas as pd
import grelu.data.preprocess as preprocess
import grelu.data.dataset
import grelu.lightning
import grelu.visualize
import pickle
from sklearn.utils import resample
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

    # 1. Filter the data by chromosome
    chr2_data = regions[regions['chrom'] == test_chroms]
    chr3_data = regions[regions['chrom'] == val_chroms]
    train_data = regions[~regions['chrom'].isin([test_chroms, val_chroms])]

    # 2. Downsample chr2 and chr3 data if necessary
    if len(chr2_data) > test_size:
        chr2_data = resample(chr2_data, n_samples=test_size, random_state=1)

    if len(chr3_data) > valid_size:
        chr3_data = resample(chr3_data, n_samples=valid_size, random_state=1)

    # 3. Downsample the training data to 12,000 if necessary
    if len(train_data) > train_size:
        train_data = resample(train_data, n_samples=train_size, random_state=1)

    # 4. Combine the final training, validation, and test sets
    train = train_data
    val = chr3_data
    test = chr2_data

    # 5. Print the sizes of each split to verify
    print("Training set size:", len(train))
    print("Validation set size:", len(val))
    print("Test set size:", len(test))
    

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
    
    with open('./datasets/unsplit_test/train_ds_compare.pkl', 'wb') as f:
        pickle.dump(train_ds, f)
        
    
    with open('./datasets/unsplit_test/val_ds_compare.pkl', 'wb') as f:
        pickle.dump(val_ds, f)
    
    
    with open('./datasets/unsplit_test/test_ds_compare.pkl', 'wb') as f:
        pickle.dump(test_ds, f)
        
    

if __name__ == '__main__':
    main()