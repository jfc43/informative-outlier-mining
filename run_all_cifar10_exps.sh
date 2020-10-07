#!/bin/bash

# train an vanilla model
python train.py --name vanilla

# train an SOFL model
python train_sofl.py --name SOFL

# train an OE model
python train_oe.py --name OE

# train an ACET model
python train_acet.py --name ACET

# train an CCU model
python train_ccu.py --name CCU

# train an ROWL model
python train_rowl.py --name ROWL

# train an NTOM model
python train_ntom.py --name NTOM

# train an ATOM model
python train_atom.py --name ATOM

# Evaluate MSP
python eval_ood_detection.py --name vanilla --method msp
python eval_ood_detection.py --name vanilla --method msp --corrupt
python eval_ood_detection.py --name vanilla --method msp --adv
python eval_ood_detection.py --name vanilla --method msp --adv-corrupt

# Evaluate ODIN:
python eval_ood_detection.py --name vanilla --method odin
python eval_ood_detection.py --name vanilla --method odin --corrupt
python eval_ood_detection.py --name vanilla --method odin --adv
python eval_ood_detection.py --name vanilla --method odin --adv-corrupt

# Evaluate Mahalanobis:
python tune_mahalanobis_hyperparams.py --name vanilla
python eval_ood_detection.py --name vanilla --method mahalanobis
python eval_ood_detection.py --name vanilla --method mahalanobis --corrupt
python eval_ood_detection.py --name vanilla --method mahalanobis --adv
python eval_ood_detection.py --name vanilla --method mahalanobis --adv-corrupt

# Evaluate SOFL:
python eval_ood_detection.py --name SOFL --method sofl
python eval_ood_detection.py --name SOFL --method sofl --corrupt
python eval_ood_detection.py --name SOFL --method sofl --adv
python eval_ood_detection.py --name SOFL --method sofl --adv-corrupt

# Evaluate OE:
python eval_ood_detection.py --name OE --method msp
python eval_ood_detection.py --name OE --method msp --corrupt
python eval_ood_detection.py --name OE --method msp --adv
python eval_ood_detection.py --name OE --method msp --adv-corrupt

# Evaluate ACET:
python eval_ood_detection.py --name ACET --method msp
python eval_ood_detection.py --name ACET --method msp --corrupt
python eval_ood_detection.py --name ACET --method msp --adv
python eval_ood_detection.py --name ACET --method msp --adv-corrupt

# Evaluate CCU:
python eval_ood_detection.py --name CCU --method msp
python eval_ood_detection.py --name CCU --method msp --corrupt
python eval_ood_detection.py --name CCU --method msp --adv
python eval_ood_detection.py --name CCU --method msp --adv-corrupt

# Evaluate ROWL:
python eval_ood_detection.py --name ROWL --method rowl
python eval_ood_detection.py --name ROWL --method rowl --corrupt
python eval_ood_detection.py --name ROWL --method rowl --adv
python eval_ood_detection.py --name ROWL --method rowl --adv-corrupt

# Evaluate NTOM:
python eval_ood_detection.py --name NTOM --method ntom
python eval_ood_detection.py --name NTOM --method ntom --corrupt
python eval_ood_detection.py --name NTOM --method ntom --adv
python eval_ood_detection.py --name NTOM --method ntom --adv-corrupt

# Evaluate ATOM:
python eval_ood_detection.py --name ATOM --method atom
python eval_ood_detection.py --name ATOM --method atom --corrupt
python eval_ood_detection.py --name ATOM --method atom --adv
python eval_ood_detection.py --name ATOM --method atom --adv-corrupt

# Compute metrics:
python compute_metrics.py --name vanilla --method msp
python compute_metrics.py --name vanilla --method odin
python compute_metrics.py --name vanilla --method mahalanobis
python compute_metrics.py --name SOFL --method sofl
python compute_metrics.py --name OE --method msp
python compute_metrics.py --name ACET --method msp
python compute_metrics.py --name CCU --method msp
python compute_metrics.py --name ROWL --method rowl
python compute_metrics.py --name NTOM --method ntom
python compute_metrics.py --name ATOM --method atom