#!/bin/bash

DATADIR="/storage/brno3-cerit/home/vojtechouska"

# nacteni aplikacniho modulu, ktery zpristupni aplikaci Gaussian verze 3

module load tensorflow-2.0.0-gpu-python3

# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)

cp -r "$DATADIR/nsynth-test" "$SCRATCHDIR"

#cp -r "$DATADIR/nsynth-valid" "$SCRATCHDIR"

#cp -r "$DATADIR/nsynth-train" "$SCRATCHDIR"

cp "$DATADIR/soundnet_clasifier/sound8.npy" "$SCRATCHDIR"

cp "$DATADIR/soundnet_clasifier/soundnet.py" "$SCRATCHDIR"

python soundnet.py

# kopirovani vysledku do DATADIR

cp "$SCRATCHDIR/activations.npy" "$DATADIR/soundnet_clasifier"
 
cp "$SCRATCHDIR/labels.npy" "$DATADIR/soundnet_clasifier"


