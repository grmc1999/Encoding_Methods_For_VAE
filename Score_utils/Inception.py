
# Load model
# Predict batch plus softmax # [batch channel width height] -> [batch classes=1000]
# take set of batches # [splits*batch classes=1000]
# take mean of splits # [splits*batch classes=1000]
# for each part take entropy between mean and ith part