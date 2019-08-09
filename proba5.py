from scipy.special import softmax
import numpy as np


def log_softmax_with_xentropy(logits, labels):
    print("labels:", labels, sep="\n")
    print()
    print("all logits:", logits, sep="\n")
    print()
    # logits = logits[:, 1]
    # print("truth logits:", logits, sep="\n")

    probs = softmax(logits, axis=1)
    print("softmax:", probs, sep="\n")
    print()

    truth_probs = []
    for row in range(probs.shape[0]):
        truth_probs.append(probs[row, labels[row]])
    truth_probs = np.array(truth_probs)

    # truth_probs = probs[:, labels]
    print("softmax for class 1", truth_probs)
    print()

    log_probs = np.log(truth_probs)
    print("log softmax:", log_probs, sep="\n")
    print()

    loss = np.sum(log_probs)
    print("sum of neg log probs", -loss, sep="\n")
    print()

    mean_loss = np.mean(log_probs)
    print("mean of neg log probs", -mean_loss, sep="\n")


logits = np.array([[  20.2123,  -21.3646],
        [-158.2895,  154.9364],
        [  57.2730,  -57.1633]])
# print("all log"logits)

labels = np.array([1, 0, 0])

log_softmax_with_xentropy(logits, labels)
