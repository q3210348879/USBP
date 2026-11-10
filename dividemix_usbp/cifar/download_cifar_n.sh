#!/bin/bash
# Download CIFAR-N human annotation noise labels
# Source: http://www.noisylabels.com / https://github.com/UCSC-REAL/cifar-10-100n

mkdir -p cifar-n-labels

# CIFAR-10N: contains 'worse_label', 'aggre_label', 'random_label1/2/3' keys
wget -O cifar-n-labels/CIFAR-10_human.pt \
  "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"

# CIFAR-100N: contains 'noisy_label' key (fine-grained)
wget -O cifar-n-labels/CIFAR-100_human.pt \
  "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt"

echo "Done. Files saved to cifar-n-labels/"
echo ""
echo "CIFAR-10N keys: worse_label, aggre_label, random_label1, random_label2, random_label3"
echo "CIFAR-100N keys: noisy_label"
