# Latest Adversarial Attack Papers
**update at 2021-12-10 23:56:46**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Mutual Adversarial Training: Learning together is better than going alone**

cs.LG

Under submission

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.05005v1)

**Authors**: Jiang Liu, Chun Pong Lau, Hossein Souri, Soheil Feizi, Rama Chellappa

**Abstracts**: Recent studies have shown that robustness to adversarial attacks can be transferred across networks. In other words, we can make a weak model more robust with the help of a strong teacher model. We ask if instead of learning from a static teacher, can models "learn together" and "teach each other" to achieve better robustness? In this paper, we study how interactions among models affect robustness via knowledge distillation. We propose mutual adversarial training (MAT), in which multiple models are trained together and share the knowledge of adversarial examples to achieve improved robustness. MAT allows robust models to explore a larger space of adversarial samples, and find more robust feature spaces and decision boundaries. Through extensive experiments on CIFAR-10 and CIFAR-100, we demonstrate that MAT can effectively improve model robustness and outperform state-of-the-art methods under white-box attacks, bringing $\sim$8% accuracy gain to vanilla adversarial training (AT) under PGD-100 attacks. In addition, we show that MAT can also mitigate the robustness trade-off among different perturbation types, bringing as much as 13.1% accuracy gain to AT baselines against the union of $l_\infty$, $l_2$ and $l_1$ attacks. These results show the superiority of the proposed method and demonstrate that collaborative learning is an effective strategy for designing robust models.



## **2. FCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack**

cs.CV

9 pages, 5 figures

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2109.07193v2)

**Authors**: Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Xiaoya Zhang, Zhiqiang Gong, Wen Yao, Xiaoqian Chen

**Abstracts**: Physical adversarial attacks in object detection have attracted increasing attention. However, most previous works focus on hiding the objects from the detector by generating an individual adversarial patch, which only covers the planar part of the vehicle's surface and fails to attack the detector in physical scenarios for multi-view, long-distance and partially occluded objects. To bridge the gap between digital attacks and physical attacks, we exploit the full 3D vehicle surface to propose a robust Full-coverage Camouflage Attack (FCA) to fool detectors. Specifically, we first try rendering the nonplanar camouflage texture over the full vehicle surface. To mimic the real-world environment conditions, we then introduce a transformation function to transfer the rendered camouflaged vehicle into a photo realistic scenario. Finally, we design an efficient loss function to optimize the camouflage texture. Experiments show that the full-coverage camouflage attack can not only outperform state-of-the-art methods under various test cases but also generalize to different environments, vehicles, and object detectors. The code of FCA will be available at: https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/.



## **3. Adversarial Attacks on Neural Networks for Graph Data**

stat.ML

Accepted as a full paper at KDD 2018 on May 6, 2018

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/1805.07984v4)

**Authors**: Daniel Zügner, Amir Akbarnejad, Stephan Günnemann

**Abstracts**: Deep learning models for graphs have achieved strong performance for the task of node classification. Despite their proliferation, currently there is no study of their robustness to adversarial attacks. Yet, in domains where they are likely to be used, e.g. the web, adversaries are common. Can deep learning models for graphs be easily fooled? In this work, we introduce the first study of adversarial attacks on attributed graphs, specifically focusing on models exploiting ideas of graph convolutions. In addition to attacks at test time, we tackle the more challenging class of poisoning/causative attacks, which focus on the training phase of a machine learning model. We generate adversarial perturbations targeting the node's features and the graph structure, thus, taking the dependencies between instances in account. Moreover, we ensure that the perturbations remain unnoticeable by preserving important data characteristics. To cope with the underlying discrete domain we propose an efficient algorithm Nettack exploiting incremental computations. Our experimental study shows that accuracy of node classification significantly drops even when performing only few perturbations. Even more, our attacks are transferable: the learned attacks generalize to other state-of-the-art node classification models and unsupervised approaches, and likewise are successful even when only limited knowledge about the graph is given.



## **4. PARL: Enhancing Diversity of Ensemble Networks to Resist Adversarial Attacks via Pairwise Adversarially Robust Loss Function**

cs.LG

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.04948v1)

**Authors**: Manaar Alam, Shubhajit Datta, Debdeep Mukhopadhyay, Arijit Mondal, Partha Pratim Chakrabarti

**Abstracts**: The security of Deep Learning classifiers is a critical field of study because of the existence of adversarial attacks. Such attacks usually rely on the principle of transferability, where an adversarial example crafted on a surrogate classifier tends to mislead the target classifier trained on the same dataset even if both classifiers have quite different architecture. Ensemble methods against adversarial attacks demonstrate that an adversarial example is less likely to mislead multiple classifiers in an ensemble having diverse decision boundaries. However, recent ensemble methods have either been shown to be vulnerable to stronger adversaries or shown to lack an end-to-end evaluation. This paper attempts to develop a new ensemble methodology that constructs multiple diverse classifiers using a Pairwise Adversarially Robust Loss (PARL) function during the training procedure. PARL utilizes gradients of each layer with respect to input in every classifier within the ensemble simultaneously. The proposed training procedure enables PARL to achieve higher robustness against black-box transfer attacks compared to previous ensemble methods without adversely affecting the accuracy of clean examples. We also evaluate the robustness in the presence of white-box attacks, where adversarial examples are crafted using parameters of the target classifier. We present extensive experiments using standard image classification datasets like CIFAR-10 and CIFAR-100 trained using standard ResNet20 classifier against state-of-the-art adversarial attacks to demonstrate the robustness of the proposed ensemble methodology.



## **5. Detecting Adversaries, yet Faltering to Noise? Leveraging Conditional Variational AutoEncoders for Adversary Detection in the Presence of Noisy Images**

cs.LG

Accepted at Adversarial Machine Learning (AdvML) workshop, AAAI 2022

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2111.15518v2)

**Authors**: Dvij Kalaria, Aritra Hazra, Partha Pratim Chakrabarti

**Abstracts**: With the rapid advancement and increased use of deep learning models in image identification, security becomes a major concern to their deployment in safety-critical systems. Since the accuracy and robustness of deep learning models are primarily attributed from the purity of the training samples, therefore the deep learning architectures are often susceptible to adversarial attacks. Adversarial attacks are often obtained by making subtle perturbations to normal images, which are mostly imperceptible to humans, but can seriously confuse the state-of-the-art machine learning models. What is so special in the slightest intelligent perturbations or noise additions over normal images that it leads to catastrophic classifications by the deep neural networks? Using statistical hypothesis testing, we find that Conditional Variational AutoEncoders (CVAE) are surprisingly good at detecting imperceptible image perturbations. In this paper, we show how CVAEs can be effectively used to detect adversarial attacks on image classification networks. We demonstrate our results over MNIST, CIFAR-10 dataset and show how our method gives comparable performance to the state-of-the-art methods in detecting adversaries while not getting confused with noisy images, where most of the existing methods falter.



## **6. On the privacy-utility trade-off in differentially private hierarchical text classification**

cs.CR

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2103.02895v2)

**Authors**: Dominik Wunderlich, Daniel Bernau, Francesco Aldà, Javier Parra-Arnau, Thorsten Strufe

**Abstracts**: Hierarchical text classification consists in classifying text documents into a hierarchy of classes and sub-classes. Although artificial neural networks have proved useful to perform this task, unfortunately they can leak training data information to adversaries due to training data memorization. Using differential privacy during model training can mitigate leakage attacks against trained models, enabling the models to be shared safely at the cost of reduced model accuracy. This work investigates the privacy-utility trade-off in hierarchical text classification with differential privacy guarantees, and identifies neural network architectures that offer superior trade-offs. To this end, we use a white-box membership inference attack to empirically assess the information leakage of three widely used neural network architectures. We show that large differential privacy parameters already suffice to completely mitigate membership inference attacks, thus resulting only in a moderate decrease in model utility. More specifically, for large datasets with long texts we observed Transformer-based models to achieve an overall favorable privacy-utility trade-off, while for smaller datasets with shorter texts convolutional neural networks are preferable.



## **7. Amicable Aid: Turning Adversarial Attack to Benefit Classification**

cs.CV

16 pages (3 pages for appendix)

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.04720v1)

**Authors**: Juyeop Kim, Jun-Ho Choi, Soobeom Jang, Jong-Seok Lee

**Abstracts**: While adversarial attacks on deep image classification models pose serious security concerns in practice, this paper suggests a novel paradigm where the concept of adversarial attacks can benefit classification performance, which we call amicable aid. We show that by taking the opposite search direction of perturbation, an image can be converted to another yielding higher confidence by the classification model and even a wrongly classified image can be made to be correctly classified. Furthermore, with a large amount of perturbation, an image can be made unrecognizable by human eyes, while it is correctly recognized by the model. The mechanism of the amicable aid is explained in the viewpoint of the underlying natural image manifold. We also consider universal amicable perturbations, i.e., a fixed perturbation can be applied to multiple images to improve their classification results. While it is challenging to find such perturbations, we show that making the decision boundary as perpendicular to the image manifold as possible via training with modified data is effective to obtain a model for which universal amicable perturbations are more easily found. Finally, we discuss several application scenarios where the amicable aid can be useful, including secure image communication, privacy-preserving image communication, and protection against adversarial attacks.



## **8. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

cs.CV

Under submission

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04532v1)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detecting and removing adversarial patches. We first train a patch segmenter that outputs patch masks that provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images given the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no performance drop on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks.



## **9. On anti-stochastic properties of unlabeled graphs**

cs.DM

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04395v1)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstracts**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.



## **10. SNEAK: Synonymous Sentences-Aware Adversarial Attack on Natural Language Video Localization**

cs.CV

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04154v1)

**Authors**: Wenbo Gou, Wen Shi, Jian Lou, Lijie Huang, Pan Zhou, Ruixuan Li

**Abstracts**: Natural language video localization (NLVL) is an important task in the vision-language understanding area, which calls for an in-depth understanding of not only computer vision and natural language side alone, but more importantly the interplay between both sides. Adversarial vulnerability has been well-recognized as a critical security issue of deep neural network models, which requires prudent investigation. Despite its extensive yet separated studies in video and language tasks, current understanding of the adversarial robustness in vision-language joint tasks like NLVL is less developed. This paper therefore aims to comprehensively investigate the adversarial robustness of NLVL models by examining three facets of vulnerabilities from both attack and defense aspects. To achieve the attack goal, we propose a new adversarial attack paradigm called synonymous sentences-aware adversarial attack on NLVL (SNEAK), which captures the cross-modality interplay between the vision and language sides.



## **11. Adversarial Prefetch: New Cross-Core Cache Side Channel Attacks**

cs.CR

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2110.12340v2)

**Authors**: Yanan Guo, Andrew Zigerelli, Youtao Zhang, Jun Yang

**Abstracts**: Modern x86 processors have many prefetch instructions that can be used by programmers to boost performance. However, these instructions may also cause security problems. In particular, we found that on Intel processors, there are two security flaws in the implementation of PREFETCHW, an instruction for accelerating future writes. First, this instruction can execute on data with read-only permission. Second, the execution time of this instruction leaks the current coherence state of the target data.   Based on these two design issues, we build two cross-core private cache attacks that work with both inclusive and non-inclusive LLCs, named Prefetch+Reload and Prefetch+Prefetch. We demonstrate the significance of our attacks in different scenarios. First, in the covert channel case, Prefetch+Reload and Prefetch+Prefetch achieve 782 KB/s and 822 KB/s channel capacities, when using only one shared cache line between the sender and receiver, the largest-to-date single-line capacities for CPU cache covert channels. Further, in the side channel case, our attacks can monitor the access pattern of the victim on the same processor, with almost zero error rate. We show that they can be used to leak private information of real-world applications such as cryptographic keys. Finally, our attacks can be used in transient execution attacks in order to leak more secrets within the transient window than prior work. From the experimental results, our attacks allow leaking about 2 times as many secret bytes, compared to Flush+Reload, which is widely used in transient execution attacks.



## **12. Two Coupled Rejection Metrics Can Tell Adversarial Examples Apart**

cs.LG

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2105.14785v3)

**Authors**: Tianyu Pang, Huishuai Zhang, Di He, Yinpeng Dong, Hang Su, Wei Chen, Jun Zhu, Tie-Yan Liu

**Abstracts**: Correctly classifying adversarial examples is an essential but challenging requirement for safely deploying machine learning models. As reported in RobustBench, even the state-of-the-art adversarially trained models struggle to exceed 67% robust test accuracy on CIFAR-10, which is far from practical. A complementary way towards robustness is to introduce a rejection option, allowing the model to not return predictions on uncertain inputs, where confidence is a commonly used certainty proxy. Along with this routine, we find that confidence and a rectified confidence (R-Con) can form two coupled rejection metrics, which could provably distinguish wrongly classified inputs from correctly classified ones. This intriguing property sheds light on using coupling strategies to better detect and reject adversarial examples. We evaluate our rectified rejection (RR) module on CIFAR-10, CIFAR-10-C, and CIFAR-100 under several attacks including adaptive ones, and demonstrate that the RR module is compatible with different adversarial training frameworks on improving robustness, with little extra computation. The code is available at https://github.com/P2333/Rectified-Rejection.



## **13. Local Convolutions Cause an Implicit Bias towards High Frequency Adversarial Examples**

stat.ML

20 pages, 11 figures, 12 Tables

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2006.11440v4)

**Authors**: Josue Ortega Caro, Yilong Ju, Ryan Pyle, Sourav Dey, Wieland Brendel, Fabio Anselmi, Ankit Patel

**Abstracts**: Adversarial Attacks are still a significant challenge for neural networks. Recent work has shown that adversarial perturbations typically contain high-frequency features, but the root cause of this phenomenon remains unknown. Inspired by theoretical work on linear full-width convolutional models, we hypothesize that the local (i.e. bounded-width) convolutional operations commonly used in current neural networks are implicitly biased to learn high frequency features, and that this is one of the root causes of high frequency adversarial examples. To test this hypothesis, we analyzed the impact of different choices of linear and nonlinear architectures on the implicit bias of the learned features and the adversarial perturbations, in both spatial and frequency domains. We find that the high-frequency adversarial perturbations are critically dependent on the convolution operation because the spatially-limited nature of local convolutions induces an implicit bias towards high frequency features. The explanation for the latter involves the Fourier Uncertainty Principle: a spatially-limited (local in the space domain) filter cannot also be frequency-limited (local in the frequency domain). Furthermore, using larger convolution kernel sizes or avoiding convolutions (e.g. by using Vision Transformers architecture) significantly reduces this high frequency bias, but not the overall susceptibility to attacks. Looking forward, our work strongly suggests that understanding and controlling the implicit bias of architectures will be essential for achieving adversarial robustness.



## **14. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

14 pages for the main text

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2009.04131v4)

**Authors**: Linyi Li, Xiangyu Qi, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.



## **15. Saliency Diversified Deep Ensemble for Robustness to Adversaries**

cs.CV

Accepted to AAAI Workshop on Adversarial Machine Learning and Beyond  2022

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03615v1)

**Authors**: Alex Bogun, Dimche Kostadinov, Damian Borth

**Abstracts**: Deep learning models have shown incredible performance on numerous image recognition, classification, and reconstruction tasks. Although very appealing and valuable due to their predictive capabilities, one common threat remains challenging to resolve. A specifically trained attacker can introduce malicious input perturbations to fool the network, thus causing potentially harmful mispredictions. Moreover, these attacks can succeed when the adversary has full access to the target model (white-box) and even when such access is limited (black-box setting). The ensemble of models can protect against such attacks but might be brittle under shared vulnerabilities in its members (attack transferability). To that end, this work proposes a novel diversity-promoting learning approach for the deep ensembles. The idea is to promote saliency map diversity (SMD) on ensemble members to prevent the attacker from targeting all ensemble members at once by introducing an additional term in our learning objective. During training, this helps us minimize the alignment between model saliencies to reduce shared member vulnerabilities and, thus, increase ensemble robustness to adversaries. We empirically show a reduced transferability between ensemble members and improved performance compared to the state-of-the-art ensemble defense against medium and high strength white-box attacks. In addition, we demonstrate that our approach combined with existing methods outperforms state-of-the-art ensemble algorithms for defense under white-box and black-box attacks.



## **16. Membership Inference Attacks From First Principles**

cs.CR

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03570v1)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.



## **17. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

cs.CV

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03492v1)

**Authors**: Yucheng Shi, Yahong Han

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Deep Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the existing decision-based attacks for CNNs ignore the difference in noise sensitivity between different regions of the image, which affects the efficiency of noise compression. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we propose a new decision-based black-box attack against ViTs termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on ImageNet-21k, ILSVRC-2012, and Tiny-Imagenet datasets demonstrate that PAR achieves a much lower magnitude of perturbation on average with the same number of queries.



## **18. BDFA: A Blind Data Adversarial Bit-flip Attack on Deep Neural Networks**

cs.CR

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03477v1)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Zhenman Fang, Lesley Shannon

**Abstracts**: Adversarial bit-flip attack (BFA) on Neural Network weights can result in catastrophic accuracy degradation by flipping a very small number of bits. A major drawback of prior bit flip attack techniques is their reliance on test data. This is frequently not possible for applications that contain sensitive or proprietary data. In this paper, we propose Blind Data Adversarial Bit-flip Attack (BDFA), a novel technique to enable BFA without any access to the training or testing data. This is achieved by optimizing for a synthetic dataset, which is engineered to match the statistics of batch normalization across different layers of the network and the targeted label. Experimental results show that BDFA could decrease the accuracy of ResNet50 significantly from 75.96\% to 13.94\% with only 4 bits flips.



## **19. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

cs.CV

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2104.14528v5)

**Authors**: Haoyuan Chen, Chen Li, Xiaoyan Li, Ge Wang, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Yudong Yao, Yueyang Teng, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.



## **20. Introducing the DOME Activation Functions**

cs.LG

16 pages, 9 figures

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2109.14798v2)

**Authors**: Mohamed E. Hussein, Wael AbdAlmageed

**Abstracts**: In this paper, we introduce a novel non-linear activation function that spontaneously induces class-compactness and regularization in the embedding space of neural networks. The function is dubbed DOME for Difference Of Mirrored Exponential terms. The basic form of the function can replace the sigmoid or the hyperbolic tangent functions as an output activation function for binary classification problems. The function can also be extended to the case of multi-class classification, and used as an alternative to the standard softmax function. It can also be further generalized to take more flexible shapes suitable for intermediate layers of a network. We empirically demonstrate the properties of the function. We also show that models using the function exhibit extra robustness against adversarial attacks.



## **21. Adversarial Attacks in Cooperative AI**

cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.14833v2)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent work in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making incorrect decisions. In addition, some past research in cooperative AI has relied on new notions of representations, like public beliefs, to accelerate the learning of optimally cooperative behavior. Hence, cooperative AI might introduce new weaknesses not investigated in previous machine learning research. In this paper, our contributions include: (1) arguing that three algorithms inspired by human-like social intelligence introduce new vulnerabilities, unique to cooperative AI, that adversaries can exploit, and (2) an experiment showing that simple, adversarial perturbations on the agents' beliefs can negatively impact performance. This evidence points to the possibility that formal representations of social behavior are vulnerable to adversarial attacks.



## **22. Shape Defense Against Adversarial Attacks**

cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2008.13336v3)

**Authors**: Ali Borji

**Abstracts**: Humans rely heavily on shape information to recognize objects. Conversely, convolutional neural networks (CNNs) are biased more towards texture. This is perhaps the main reason why CNNs are vulnerable to adversarial examples. Here, we explore how shape bias can be incorporated into CNNs to improve their robustness. Two algorithms are proposed, based on the observation that edges are invariant to moderate imperceptible perturbations. In the first one, a classifier is adversarially trained on images with the edge map as an additional channel. At inference time, the edge map is recomputed and concatenated to the image. In the second algorithm, a conditional GAN is trained to translate the edge maps, from clean and/or perturbed images, into clean images. Inference is done over the generated image corresponding to the input's edge map. Extensive experiments over 10 datasets demonstrate the effectiveness of the proposed algorithms against FGSM and $\ell_\infty$ PGD-40 attacks. Further, we show that a) edge information can also benefit other adversarial training methods, and b) CNNs trained on edge-augmented inputs are more robust against natural image corruptions such as motion blur, impulse noise and JPEG compression, than CNNs trained solely on RGB images. From a broader perspective, our study suggests that CNNs do not adequately account for image structures that are crucial for robustness. Code is available at:~\url{https://github.com/aliborji/Shapedefense.git}.



## **23. Adversarial Machine Learning In Network Intrusion Detection Domain: A Systematic Review**

cs.CR

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03315v1)

**Authors**: Huda Ali Alatwi, Charles Morisset

**Abstracts**: Due to their massive success in various domains, deep learning techniques are increasingly used to design network intrusion detection solutions that detect and mitigate unknown and known attacks with high accuracy detection rates and minimal feature engineering. However, it has been found that deep learning models are vulnerable to data instances that can mislead the model to make incorrect classification decisions so-called (adversarial examples). Such vulnerability allows attackers to target NIDSs by adding small crafty perturbations to the malicious traffic to evade detection and disrupt the system's critical functionalities. The problem of deep adversarial learning has been extensively studied in the computer vision domain; however, it is still an area of open research in network security applications. Therefore, this survey explores the researches that employ different aspects of adversarial machine learning in the area of network intrusion detection in order to provide directions for potential solutions. First, the surveyed studies are categorized based on their contribution to generating adversarial examples, evaluating the robustness of ML-based NIDs towards adversarial examples, and defending these models against such attacks. Second, we highlight the characteristics identified in the surveyed research. Furthermore, we discuss the applicability of the existing generic adversarial attacks for the NIDS domain, the feasibility of launching the proposed attacks in real-world scenarios, and the limitations of the existing mitigation solutions.



## **24. Context-Aware Transfer Attacks for Object Detection**

cs.CV

accepted to AAAI 2022

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03223v1)

**Authors**: Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to $20$ percentage points improvement in performance compared to the other state-of-the-art methods.



## **25. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

cs.SD

Accepted by TASLP

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2106.00273v3)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.



## **26. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

cs.CV

To be published on Artificial Intelligence Review journal (after  minor revision)

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2105.00203v3)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.



## **27. Robust Person Re-identification with Multi-Modal Joint Defence**

cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.09571v2)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.



## **28. ML Attack Models: Adversarial Attacks and Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02797v1)

**Authors**: Jing Lin, Long Dang, Mohamed Rahouti, Kaiqi Xiong

**Abstracts**: Many state-of-the-art ML models have outperformed humans in various tasks such as image classification. With such outstanding performance, ML models are widely used today. However, the existence of adversarial attacks and data poisoning attacks really questions the robustness of ML models. For instance, Engstrom et al. demonstrated that state-of-the-art image classifiers could be easily fooled by a small rotation on an arbitrary image. As ML systems are being increasingly integrated into safety and security-sensitive applications, adversarial attacks and data poisoning attacks pose a considerable threat. This chapter focuses on the two broad and important areas of ML security: adversarial attacks and data poisoning attacks.



## **29. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v4)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.



## **30. Staring Down the Digital Fulda Gap Path Dependency as a Cyber Defense Vulnerability**

cs.CY

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02773v1)

**Authors**: Jan Kallberg

**Abstracts**: Academia, homeland security, defense, and media have accepted the perception that critical infrastructure in a future cyber war cyber conflict is the main gateway for a massive cyber assault on the U.S. The question is not if the assumption is correct or not, the question is instead of how did we arrive at that assumption. The cyber paradigm considers critical infrastructure the primary attack vector for future cyber conflicts. The national vulnerability embedded in critical infrastructure is given a position in the cyber discourse as close to an unquestionable truth as a natural law.   The American reaction to Sept. 11, and any attack on U.S. soil, hint to an adversary that attacking critical infrastructure to create hardship for the population could work contrary to the intended softening of the will to resist foreign influence. It is more likely that attacks that affect the general population instead strengthen the will to resist and fight, similar to the British reaction to the German bombing campaign Blitzen in 1940. We cannot rule out attacks that affect the general population, but there are not enough adversarial offensive capabilities to attack all 16 critical infrastructure sectors and gain strategic momentum. An adversary has limited cyberattack capabilities and needs to prioritize cyber targets that are aligned with the overall strategy. Logically, an adversary will focus their OCO on operations that has national security implications and support their military operations by denying, degrading, and confusing the U.S. information environment and U.S. cyber assets.



## **31. Label-Only Membership Inference Attacks**

cs.CR

16 pages, 11 figures, 2 tables Revision 2: 19 pages, 12 figures, 3  tables. Improved text and additional experiments. Final ICML paper

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2007.14321v3)

**Authors**: Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, Nicolas Papernot

**Abstracts**: Membership inference attacks are one of the simplest forms of privacy leakage for machine learning models: given a data point and model, determine whether the point was used to train the model. Existing membership inference attacks exploit models' abnormal confidence when queried on their training data. These attacks do not apply if the adversary only gets access to models' predicted labels, without a confidence measure. In this paper, we introduce label-only membership inference attacks. Instead of relying on confidence scores, our attacks evaluate the robustness of a model's predicted labels under perturbations to obtain a fine-grained membership signal. These perturbations include common data augmentations or adversarial examples. We empirically show that our label-only membership inference attacks perform on par with prior attacks that required access to model confidences. We further demonstrate that label-only attacks break multiple defenses against membership inference attacks that (implicitly or explicitly) rely on a phenomenon we call confidence masking. These defenses modify a model's confidence scores in order to thwart attacks, but leave the model's predicted labels unchanged. Our label-only attacks demonstrate that confidence-masking is not a viable defense strategy against membership inference. Finally, we investigate worst-case label-only attacks, that infer membership for a small number of outlier data points. We show that label-only attacks also match confidence-based attacks in this setting. We find that training models with differential privacy and (strong) L2 regularization are the only known defense strategies that successfully prevents all attacks. This remains true even when the differential privacy budget is too high to offer meaningful provable guarantees.



## **32. Learning Swarm Interaction Dynamics from Density Evolution**

eess.SY

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02675v1)

**Authors**: Christos Mavridis, Amoolya Tirumalai, John Baras

**Abstracts**: We consider the problem of understanding the coordinated movements of biological or artificial swarms. In this regard, we propose a learning scheme to estimate the coordination laws of the interacting agents from observations of the swarm's density over time. We describe the dynamics of the swarm based on pairwise interactions according to a Cucker-Smale flocking model, and express the swarm's density evolution as the solution to a system of mean-field hydrodynamic equations. We propose a new family of parametric functions to model the pairwise interactions, which allows for the mean-field macroscopic system of integro-differential equations to be efficiently solved as an augmented system of PDEs. Finally, we incorporate the augmented system in an iterative optimization scheme to learn the dynamics of the interacting agents from observations of the swarm's density evolution over time. The results of this work can offer an alternative approach to study how animal flocks coordinate, create new control schemes for large networked systems, and serve as a central part of defense mechanisms against adversarial drone attacks.



## **33. Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness**

cs.LG

Bayesian Deep Learning Workshop, NeurIPS 2021

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02671v1)

**Authors**: Konstantinos P. Panousis, Sotirios Chatzis, Sergios Theodoridis

**Abstracts**: This work explores the potency of stochastic competition-based activations, namely Stochastic Local Winner-Takes-All (LWTA), against powerful (gradient-based) white-box and black-box adversarial attacks; we especially focus on Adversarial Training settings. In our work, we replace the conventional ReLU-based nonlinearities with blocks comprising locally and stochastically competing linear units. The output of each network layer now yields a sparse output, depending on the outcome of winner sampling in each block. We rely on the Variational Bayesian framework for training and inference; we incorporate conventional PGD-based adversarial training arguments to increase the overall adversarial robustness. As we experimentally show, the arising networks yield state-of-the-art robustness against powerful adversarial attacks while retaining very high classification rate in the benign case.



## **34. Formalizing and Estimating Distribution Inference Risks**

cs.LG

Shorter version of work available at arXiv:2106.03699 Update: New  version with more theoretical results and a deeper exploration of results

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v4)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.



## **35. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

cs.CV

9 pages

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.00428v2)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstracts**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.



## **36. Statically Detecting Adversarial Malware through Randomised Chaining**

cs.CR

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2111.14037v2)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.



## **37. Generalized Likelihood Ratio Test for Adversarially Robust Hypothesis Testing**

stat.ML

Submitted to the IEEE Transactions on Signal Processing

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.02209v1)

**Authors**: Bhagyashree Puranik, Upamanyu Madhow, Ramtin Pedarsani

**Abstracts**: Machine learning models are known to be susceptible to adversarial attacks which can cause misclassification by introducing small but well designed perturbations. In this paper, we consider a classical hypothesis testing problem in order to develop fundamental insight into defending against such adversarial perturbations. We interpret an adversarial perturbation as a nuisance parameter, and propose a defense based on applying the generalized likelihood ratio test (GLRT) to the resulting composite hypothesis testing problem, jointly estimating the class of interest and the adversarial perturbation. While the GLRT approach is applicable to general multi-class hypothesis testing, we first evaluate it for binary hypothesis testing in white Gaussian noise under $\ell_{\infty}$ norm-bounded adversarial perturbations, for which a known minimax defense optimizing for the worst-case attack provides a benchmark. We derive the worst-case attack for the GLRT defense, and show that its asymptotic performance (as the dimension of the data increases) approaches that of the minimax defense. For non-asymptotic regimes, we show via simulations that the GLRT defense is competitive with the minimax approach under the worst-case attack, while yielding a better robustness-accuracy tradeoff under weaker attacks. We also illustrate the GLRT approach for a multi-class hypothesis testing problem, for which a minimax strategy is not known, evaluating its performance under both noise-agnostic and noise-aware adversarial settings, by providing a method to find optimal noise-aware attacks, and heuristics to find noise-agnostic attacks that are close to optimal in the high SNR regime.



## **38. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01967v1)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.



## **39. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

cs.LG

In ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.01208v2)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.



## **40. Graph Neural Networks Inspired by Classical Iterative Algorithms**

cs.LG

accepted as long oral for ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.06064v4)

**Authors**: Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang, Zengfeng Huang, David Wipf

**Abstracts**: Despite the recent success of graph neural networks (GNN), common architectures often exhibit significant limitations, including sensitivity to oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur as a result of graph heterophily or adversarial attacks. To at least partially address these issues within a simple transparent framework, we consider a new family of GNN layers designed to mimic and integrate the update rules of two classical iterative algorithms, namely, proximal gradient descent and iterative reweighted least squares (IRLS). The former defines an extensible base GNN architecture that is immune to oversmoothing while nonetheless capturing long-range dependencies by allowing arbitrary propagation steps. In contrast, the latter produces a novel attention mechanism that is explicitly anchored to an underlying end-to-end energy function, contributing stability with respect to edge uncertainty. When combined we obtain an extremely simple yet robust model that we evaluate across disparate scenarios including standardized benchmarks, adversarially-perturbated graphs, graphs with heterophily, and graphs involving long-range dependencies. In doing so, we compare against SOTA GNN approaches that have been explicitly designed for the respective task, achieving competitive or superior node classification accuracy. Our code is available at https://github.com/FFTYYY/TWIRLS.



## **41. Blackbox Untargeted Adversarial Testing of Automatic Speech Recognition Systems**

cs.SD

10 pages, 6 figures and 7 tables

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01821v1)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) systems are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the correctness of ASRS, we propose techniques that automatically generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. Much of the existing work on adversarial ASR testing focuses on targeted attacks, i.e generating audio samples given an output text. Targeted techniques are not portable, customised to the structure of DNNs (whitebox) within a specific ASR. In contrast, our method attacks the signal processing stage of the ASR pipeline that is shared across most ASRs. Additionally, we ensure the generated adversarial audio samples have no human audible difference by manipulating the acoustic signal using a psychoacoustic model that maintains the signal below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and three input audio datasets using the metrics - WER of output text, Similarity to original audio and attack Success Rate on different ASRs. We found our testing techniques were portable across ASRs, with the adversarial audio samples producing high Success Rates, WERs and Similarities to the original audio.



## **42. Attack-Centric Approach for Evaluating Transferability of Adversarial Samples in Machine Learning Models**

cs.LG

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01777v1)

**Authors**: Tochukwu Idika, Ismail Akturk

**Abstracts**: Transferability of adversarial samples became a serious concern due to their impact on the reliability of machine learning system deployments, as they find their way into many critical applications. Knowing factors that influence transferability of adversarial samples can assist experts to make informed decisions on how to build robust and reliable machine learning systems. The goal of this study is to provide insights on the mechanisms behind the transferability of adversarial samples through an attack-centric approach. This attack-centric perspective interprets how adversarial samples would transfer by assessing the impact of machine learning attacks (that generated them) on a given input dataset. To achieve this goal, we generated adversarial samples using attacker models and transferred these samples to victim models. We analyzed the behavior of adversarial samples on victim models and outlined four factors that can influence the transferability of adversarial samples. Although these factors are not necessarily exhaustive, they provide useful insights to researchers and practitioners of machine learning systems.



## **43. Single-Shot Black-Box Adversarial Attacks Against Malware Detectors: A Causal Language Model Approach**

cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01724v1)

**Authors**: James Lee Hu, Mohammadreza Ebrahimi, Hsinchun Chen

**Abstracts**: Deep Learning (DL)-based malware detectors are increasingly adopted for early detection of malicious behavior in cybersecurity. However, their sensitivity to adversarial malware variants has raised immense security concerns. Generating such adversarial variants by the defender is crucial to improving the resistance of DL-based malware detectors against them. This necessity has given rise to an emerging stream of machine learning research, Adversarial Malware example Generation (AMG), which aims to generate evasive adversarial malware variants that preserve the malicious functionality of a given malware. Within AMG research, black-box method has gained more attention than white-box methods. However, most black-box AMG methods require numerous interactions with the malware detectors to generate adversarial malware examples. Given that most malware detectors enforce a query limit, this could result in generating non-realistic adversarial examples that are likely to be detected in practice due to lack of stealth. In this study, we show that a novel DL-based causal language model enables single-shot evasion (i.e., with only one query to malware detector) by treating the content of the malware executable as a byte sequence and training a Generative Pre-Trained Transformer (GPT). Our proposed method, MalGPT, significantly outperformed the leading benchmark methods on a real-world malware dataset obtained from VirusTotal, achieving over 24.51\% evasion rate. MalGPT enables cybersecurity researchers to develop advanced defense capabilities by emulating large-scale realistic AMG.



## **44. Adversarial Attacks against a Satellite-borne Multispectral Cloud Detector**

cs.CV

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01723v1)

**Authors**: Andrew Du, Yee Wei Law, Michele Sasdelli, Bo Chen, Ken Clarke, Michael Brown, Tat-Jun Chin

**Abstracts**: Data collected by Earth-observing (EO) satellites are often afflicted by cloud cover. Detecting the presence of clouds -- which is increasingly done using deep learning -- is crucial preprocessing in EO applications. In fact, advanced EO satellites perform deep learning-based cloud detection on board the satellites and downlink only clear-sky data to save precious bandwidth. In this paper, we highlight the vulnerability of deep learning-based cloud detection towards adversarial attacks. By optimising an adversarial pattern and superimposing it into a cloudless scene, we bias the neural network into detecting clouds in the scene. Since the input spectra of cloud detectors include the non-visible bands, we generated our attacks in the multispectral domain. This opens up the potential of multi-objective attacks, specifically, adversarial biasing in the cloud-sensitive bands and visual camouflage in the visible bands. We also investigated mitigation strategies against the adversarial attacks. We hope our work further builds awareness of the potential of adversarial attacks in the EO community.



## **45. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

cs.CV

AAAI-22 AdvML Workshop ShortPaper

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01601v1)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.



## **46. Is Approximation Universally Defensive Against Adversarial Attacks in Deep Neural Networks?**

cs.LG

Accepted for publication in DATE 2022

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01555v1)

**Authors**: Ayesha Siddique, Khaza Anuarul Hoque

**Abstracts**: Approximate computing is known for its effectiveness in improvising the energy efficiency of deep neural network (DNN) accelerators at the cost of slight accuracy loss. Very recently, the inexact nature of approximate components, such as approximate multipliers have also been reported successful in defending adversarial attacks on DNNs models. Since the approximation errors traverse through the DNN layers as masked or unmasked, this raises a key research question-can approximate computing always offer a defense against adversarial attacks in DNNs, i.e., are they universally defensive? Towards this, we present an extensive adversarial robustness analysis of different approximate DNN accelerators (AxDNNs) using the state-of-the-art approximate multipliers. In particular, we evaluate the impact of ten adversarial attacks on different AxDNNs using the MNIST and CIFAR-10 datasets. Our results demonstrate that adversarial attacks on AxDNNs can cause 53% accuracy loss whereas the same attack may lead to almost no accuracy loss (as low as 0.06%) in the accurate DNN. Thus, approximate computing cannot be referred to as a universal defense strategy against adversarial attacks.



## **47. FedRAD: Federated Robust Adaptive Distillation**

cs.LG

Accepted for 1st NeurIPS Workshop on New Frontiers in Federated  Learning (NFFL 2021), Virtual Meeting

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01405v1)

**Authors**: Stefán Páll Sturluson, Samuel Trew, Luis Muñoz-González, Matei Grama, Jonathan Passerat-Palmbach, Daniel Rueckert, Amir Alansary

**Abstracts**: The robustness of federated learning (FL) is vital for the distributed training of an accurate global model that is shared among large number of clients. The collaborative learning framework by typically aggregating model updates is vulnerable to model poisoning attacks from adversarial clients. Since the shared information between the global server and participants are only limited to model parameters, it is challenging to detect bad model updates. Moreover, real-world datasets are usually heterogeneous and not independent and identically distributed (Non-IID) among participants, which makes the design of such robust FL pipeline more difficult. In this work, we propose a novel robust aggregation method, Federated Robust Adaptive Distillation (FedRAD), to detect adversaries and robustly aggregate local models based on properties of the median statistic, and then performing an adapted version of ensemble Knowledge Distillation. We run extensive experiments to evaluate the proposed method against recently published works. The results show that FedRAD outperforms all other aggregators in the presence of adversaries, as well as in heterogeneous data distributions.



## **48. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

cs.AI

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01156v1)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work on constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework supports the use cases reported in the literature and can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective on two datasets from different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.



## **49. Adversarial Robustness of Deep Reinforcement Learning based Dynamic Recommender Systems**

cs.LG

arXiv admin note: text overlap with arXiv:2006.07934

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.00973v1)

**Authors**: Siyu Wang, Yuanjiang Cao, Xiaocong Chen, Lina Yao, Xianzhi Wang, Quan Z. Sheng

**Abstracts**: Adversarial attacks, e.g., adversarial perturbations of the input and adversarial samples, pose significant challenges to machine learning and deep learning techniques, including interactive recommendation systems. The latent embedding space of those techniques makes adversarial attacks difficult to detect at an early stage. Recent advance in causality shows that counterfactual can also be considered one of ways to generate the adversarial samples drawn from different distribution as the training samples. We propose to explore adversarial examples and attack agnostic detection on reinforcement learning-based interactive recommendation systems. We first craft different types of adversarial examples by adding perturbations to the input and intervening on the casual factors. Then, we augment recommendation systems by detecting potential attacks with a deep learning-based classifier based on the crafted data. Finally, we study the attack strength and frequency of adversarial examples and evaluate our model on standard datasets with multiple crafting methods. Our extensive experiments show that most adversarial attacks are effective, and both attack strength and attack frequency impact the attack performance. The strategically-timed attack achieves comparative attack performance with only 1/3 to 1/2 attack frequency. Besides, our black-box detector trained with one crafting method has the generalization ability over several other crafting methods.



## **50. Learning Task-aware Robust Deep Learning Systems**

cs.LG

9 Pages

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2010.05125v2)

**Authors**: Keji Han, Yun Li, Xianzhong Long, Yao Ge

**Abstracts**: Many works demonstrate that deep learning system is vulnerable to adversarial attack. A deep learning system consists of two parts: the deep learning task and the deep model. Nowadays, most existing works investigate the impact of the deep model on robustness of deep learning systems, ignoring the impact of the learning task. In this paper, we adopt the binary and interval label encoding strategy to redefine the classification task and design corresponding loss to improve robustness of the deep learning system. Our method can be viewed as improving the robustness of deep learning systems from both the learning task and deep model. Experimental results demonstrate that our learning task-aware method is much more robust than traditional classification while retaining the accuracy.



