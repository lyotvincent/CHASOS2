CHASOS2: A deep learning framework for chromatin loop de novo prediction with enhanced feature extraction
===
CHASOS2 (CHromatin loop prediction with Anchor Score and OCR Score), a user-friendly toolkit for de novo prediction and evaluation of chromatin loop.  
We extended the earlier work over 30%, incorporating new algorithm and toolkit as well as experimental validations. In particular, compared with the earlier publication.  
This new submission develops a new chromatin loop prediction algorithm and user-friendly De Novo prediction toolkit CHASOS2. In addition, this new submission presents a new case study and in-depth validation experiment applying CHASOS2 de novo prediction toolkit on the K562 cell line, which demonstrates high consistency with ChIA-PET identified chromatin loops

# Menu
* /experiments - the source code of comparison methods
* /figure - the source code of figures (Fig.2-4) in paper
* /source - the source code of CHASOS
  - /data_preprocess - source code of data preprocess
  - /pretrained_model - source code of anchor score model
  - /fine_tuned_ocr_model - source code of OCR score model
  - /loop_model - source code of loop prediction model
* /data - the data used in CHASOS
  * /ChIA-PET/CTCF/raw_data - raw data of detect chromatin loops used for training and testing
  * /ChIA-PET2_result - the result of chromatin loop detection by ChIA-PET2
  * /ChIP-seq - ChIP-seq data of CTCF used in chromatin loops filtering
  * /DNase - DNase data used in chromatin open region prediction in OCR score model
  * /FIMO_result - the result of motif scanning by FIMO
  * /negative_loop - the negative chromatin loops used in training and testing
  * /phastcons - sequence conservation score used in chromatin loop predictions
  * /positive_loop - the positive chromatin loops used in training and testing
  * /pretrained_data - the data used in training anchor score model
* /ref_block - some DL block tested in search space of model construction

# Data preprocess
Anchor score model dataset: `/source/data_preprocess/pretrained_data_loader.py`  
OCR score model dataset: `/source/data_preprocess/dnase_preprocessor.py`  
Loop prediction model dataset: `/source/data_preprocess/loop_preprocessor.py`, `/source/data_preprocess/positive_loop_generator.py`, `/source/data_preprocess/negative_loop_generator.py`, `/source/data_preprocess/motif_strength_adder.py`

# Anchor score model
* main training code: /source/pretrained_model/trainer.py
* model code: /source/pretrained_model/models.py - AnchorModel_v20230516_v1
* designed module in model: /source/pretrained_model/custom_layer.py
* training data preparation code: /source/pretrained_model/pretrained_data_loader.py

The anchor score model generates a single value between 0 and 1 as its output, which indicates the probability of the input sequence serving as an anchor for a loop. This output could be directly employed as a feature for predicting chromatin loops.

# OCR score model
* main training code: /source/fine_tuned_ocr_model/ocr_trainer.py
* model code: /source/fine_tuned_ocr_model/ocr_models.py - OCRModel_v20230524_v1
* designed module in model: /source/pretrained_model/custom_layer.py
* training data preparation code: /source/fine_tuned_ocr_model/ocr_data_loader.py

The OCR score model produces a [1000,1] tensor representing 1000 positions of the input sequence. This tensor signifies the predicted signal values of DNase hypersensitive site at these 1000 positions. The average of these values is computed and employed as a feature for chromatin loop prediction.

# Loop prediction model
* main training code: /source/loop_model/loop_ml_model_trainer.py
* training data preparation code: /source/loop_model/preprocess_data.py
* drawing K562 loop prediction example code (Figure 5 in paper): /source/loop_model/raw_predictor.py

In the loop prediction model section, the gradient boosting tree algo-rithm is employed. It combines the two simulated features ob-tained from the anchor score and OCR score models with the se-quence and functional genomics features commonly used for pre-diction. The gradient boosting tree outputs a single value as the prediction, which assesses the probability of constituting a chro-matin loop.

# Training environment
The anchor score model and OCR score model are trained efficiently on a NVIDIA GeForce RTX 3060 12G GPU.  
The models are implemented in a Python 3.8 environment, utilizing PyTorch 1.12 as the backend framework.

# Persistent model directory
model trained in experiment "Comparative evaluation on different chromosomes"
> OCR model: /source/fine_tuned_ocr_model/model/OCRModel_val_loss_2023_06_30_23_34_50.pt  
Anchor model: /source/pretrained_model/model/PretrainedModel_2023_06_29_20_59_01.pt  
Loop model: /source/loop_model/model/RandomForestClassifier_mine_allfeats_8cellline_bychr.joblib (It's just named a random forest, but it's actually a gradient boosting tree)

model trained in experiment "Comparative evaluation in cancer cell lines"
> OCR model: /source/fine_tuned_ocr_model/model/OCRModel_val_loss_2023_07_07_19_04_18.pt  
Anchor model: /source/pretrained_model/model/PretrainedModel_2023_07_06_20_42_59_AnchorModel_v20230516_v1.pt  
Loop model: /source/loop_model/model/RandomForestClassifier_mine_allfeats_8cellline_byhealth.joblib (It's just named a random forest, but it's actually a gradient boosting tree)


