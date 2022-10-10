# CheXRay-v2

A website for automatically diagnosing chest x-rays using generated radiologist reports and patient information.

<img width="1252" alt="Screen Shot 2022-10-10 at 2 07 17 PM" src="https://user-images.githubusercontent.com/40700820/194953166-9e1a7e5b-6cae-42cc-8ced-ecd68c7fc43e.png">
<img width="1357" alt="Screen Shot 2022-10-10 at 2 11 30 PM" src="https://user-images.githubusercontent.com/40700820/194953183-6390a2f2-1124-412e-b3c6-8ff7e540cd8d.png">
<img width="652" alt="Screen Shot 2022-10-10 at 2 11 43 PM" src="https://user-images.githubusercontent.com/40700820/194953191-5c1c1e98-5ef6-4cfe-891a-18919f5d83c8.png">
<img width="799" alt="Screen Shot 2022-10-10 at 2 13 04 PM" src="https://user-images.githubusercontent.com/40700820/194953219-436ef691-5f08-4916-a55c-9a890e2ed767.png">

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andrewhinh/CheXRay-v2/HEAD?urlpath=voila%2Frender%2Fproduction.ipynb?voila-theme=dark)

## Notes:
- Using approximately 8% of the MIMIC-CXR dataset for training, when evaluated on the official test set, the report generation model achieves a Bleu4 score of 0.0704 while the diagnosis model achieves a PR-AUC score of .7688, 54.49% precision, 82.41% recall, and an F2 score of .733. Metrics such as accuracy and ROC AUC were avoided in the optimization of the model and in their reporting because of the dataset's imbalance between positive and negative examples for each condition. A state-of-the-art model to compare these metrics to can be found [here](https://aclanthology.org/2020.emnlp-main.112.pdf).
- Inference finishes within 3-6 minutes, depending on the number of views and x-rays provided.

## Credit:
- [fast.ai](https://github.com/fastai/fastai) for their model training code.
- [Shenzhen Research Institute of Big Data](https://github.com/cuhksz-nlp/R2Gen) for their radiologist report generation model.
- [Zachary Mueller](https://github.com/muellerzr) for his ideas around multi-modal model training.
- [Binder](https://mybinder.org/) for their repository hosting platform.
