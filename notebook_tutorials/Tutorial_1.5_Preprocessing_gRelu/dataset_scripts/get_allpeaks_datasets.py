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
    dna_bins = pd.read_csv('raw_data/512bpResolution_p_values.csv.gz')  # Intervals and coverage from chromosome's 1-5 with a 512bp resolution (summed)

    # Set a threshold to filter our data
    threshold = 2
    thresholdarc = np.arcsinh(threshold) # The threshold has to undergo an arcsinh transformation as well


    # Balanced data sets
    
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
    
   
    regions = peaks
    print(regions)

    # We now have regions of peaks


    
    # Using SKLearn's resample to reduce size of validation and test sets + create data splits.
    train_size = (regions[~regions['chrom'].isin([test_chroms, val_chroms])]).shape[0]
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


    # Check final distributions of peaks is ok:
    


    ### Looking at the distribution of peaks in train_ds vs training chromosomes
    
    labels = np.array(train_ds.get_labels()).reshape(-1)
    # Reversing transformations of train_ds labels + thresholding
    thresholded_labels = np.where(np.arcsinh(np.sinh(labels) //PRED_RES)>= thresholdarc, 1, 0)
    num_ones = np.sum(thresholded_labels)
    
    print(f"% of peaks in train_ds : {num_ones*100/(len(train_ds)):.1f}%")

    # Apply the threshold to unfiltered training data
    training_chroms = dna_bins[(dna_bins['chrom'] != test_chroms) & (dna_bins['chrom'] != val_chroms)]
    thresh_label = np.where(training_chroms['cov'] >= threshold, 1, 0)
    # Calculate the number of peaks in training chromosomes
    num_peaks_train_chroms = np.sum(thresh_label)
    percentage_peaks_chroms = num_peaks_train_chroms * 100 / len(training_chroms)
    print(f"% of peaks in real distribution training chroms (1, 4, 5): {percentage_peaks_chroms:.1f}%")



    ### Looking at the distribution of peaks in val_ds vs validation chromosome
    
    labels = np.array(val_ds.get_labels()).reshape(-1)
    thresholded_labels = np.where(np.arcsinh(np.sinh(labels) //PRED_RES)>= thresholdarc, 1, 0)
    num_ones = np.sum(thresholded_labels)
    
    print(f"% of peaks in val_ds : {num_ones*100/(len(val_ds)):.1f}%")

    # Apply the threshold to unfiltered chrom3 data
    chrom3_data = dna_bins[dna_bins['chrom'] == val_chroms]
    thresh_label = np.where(chrom3_data['cov'] >= threshold, 1, 0)
    # Calculate the number of peaks in chrom3 data
    num_peaks_chrom3 = np.sum(thresh_label)
    percentage_peaks_chrom3 = num_peaks_chrom3 * 100 / len(chrom3_data)
    
    print(f"% of actual peaks in val chrom (3): {percentage_peaks_chrom3:.1f}%")



    ### Looking at the distribution of peaks in test_ds vs test chromosome
    

    # Ensuring the distribution of test_ds is similar to chromosome 2s distribution.
    labels = np.array(test_ds.get_labels()).reshape(-1)
    thresholded_labels = np.where(np.sinh(labels) //PRED_RES>= threshold, 1, 0)
    num_ones = np.sum(thresholded_labels)
    
    print(f"% of peaks in test_ds : {num_ones*100/(len(test_ds)):.1f}%")


    # Apply the threshold to chrom2_data
    chrom2_data = dna_bins[dna_bins['chrom'] == test_chroms]
    thresh_label = np.where(chrom2_data['cov'] >= threshold, 1, 0)
    # Calculate the number of peaks in chrom2_data
    num_peaks_chrom2 = np.sum(thresh_label)
    percentage_peaks_chrom2 = num_peaks_chrom2 * 100 / len(chrom2_data)
    
    print(f"% of actual peaks in test chrom (2): {percentage_peaks_chrom2:.1f}%")



    # Save data as pkl files
    
    with open('./datasets/train_ds_allpeaks.pkl', 'wb') as f:
        pickle.dump(train_ds, f)

    
    with open('./datasets/val_ds_allpeaks.pkl', 'wb') as f:
        pickle.dump(val_ds, f)


    with open('./datasets/test_ds_allpeaks.pkl', 'wb') as f:
        pickle.dump(test_ds, f)
    

if __name__ == '__main__':
    main()