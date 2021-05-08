# Flower Species Classification

- Built a two-stage framework using Deep Learning and Evolutionary Optimization to classify flower images into respective classes
- Dataset was obtained from kaggle : https://www.kaggle.com/alxmamaev/flowers-recognition
- Used Transfer Learning for deep feature extraction using a pre-trained ResNet50 model, implementation done in PyTorch
- Used Genetic Algorithm (GA) for Feature Selection on the extracted features and improved classification accuracy by around 1% along with a feature reduction of 54% (235 features selected out of 512), final feature set classification validated using KNN, SVM and RandomForestClassifier
- Achieved validation accuracy of <b>94.34%</b> from Deep Learning and <b>95.26%</b> after applying Feature Selection
