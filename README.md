# Master-Thesis-Project
# Open Set Recognition in Semantic Segmentation

  Deep Neural Networks have recently shown great success in many machine learning domains and problems. Traditional machine learning models
assume that the classes encountered during testing is always a subset of the
classes encountered during training, or in other words, a closed set. However,
this assumption is violated often as real world tasks in computer vision often come across the Open Set Recognition (OSR) problem where incomplete
knowledge of the world and several unknown inputs exists, i.e. never-beforeseen classes are introduced during testing. This requires the model to accurately classify the known classes and effectively deal with the unknown classes.

  Applying these models for safety-critical applications, such as autonomous driving and medical analysis, pose many safety challenges due to prediction uncertainty and misclassification errors. These erroneous information, can lead to poor decisions, causing accidents and casualties. Hence,
the ability to localize the unknown objects for such mission critical applications would be highly beneficial. This thesis investigates a methodology that
tackles detecting unknown objects for a semantic segmentation task.

  It builds on the intuition that, during segmentation, a network ideally
produces erroneous labels or misclassifications in regions where the unknown
objects are present. Therefore, reconstructing an image from the resulting
predicted semantic label map (which has misclassified the unknown objects),
should produce significant discrepancies with respect to the input image.
Hence, the problem of segmenting these unknown objects is reformulated as
localizing the poorly reconstructed regions in an image. The effectiveness
of the proposed method is demonstrated using two segmentation networks
evaluated on two publicly available segmentation datasets. Using the Area
Under the Receiver Operating Characteristic (AUROC) metric, it is proved
that this method can perform significantly better than standard uncertainty
estimation methods and the baseline method.
