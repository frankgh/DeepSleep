import time
import numpy
import random
from datetime import timedelta

from DeepSleepClassifier import DeepSleepClassifier

# For reproducibility
random.seed(7493)
numpy.random.seed(7493)

if __name__ == "__main__":
    start = time.time()
    print 'Setting up'

    classifier = DeepSleepClassifier('/home/afguerrerohernan/data/patients_processed/',
                                     '/home/afguerrerohernan/work/DeepSleep/exp003/',
                                     epochs=1)

    model, _ = classifier.train_model()
    classifier.test_model(model)
    print 'Training completed'
    elapsed = time.time() - start
    print 'Elapsed:', str(timedelta(seconds=elapsed))
