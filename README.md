# Latest Adversarial Attack Papers
**update at 2022-01-04 17:02:18**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Revisiting PGD Attacks for Stability Analysis of Large-Scale Nonlinear Systems and Perception-Based Control**

math.OC

Submitted to L4DC 2022

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00801v1)

**Authors**: Aaron Havens, Darioush Keivan, Peter Seiler, Geir Dullerud, Bin Hu

**Abstracts**: Many existing region-of-attraction (ROA) analysis tools find difficulty in addressing feedback systems with large-scale neural network (NN) policies and/or high-dimensional sensing modalities such as cameras. In this paper, we tailor the projected gradient descent (PGD) attack method developed in the adversarial learning community as a general-purpose ROA analysis tool for large-scale nonlinear systems and end-to-end perception-based control. We show that the ROA analysis can be approximated as a constrained maximization problem whose goal is to find the worst-case initial condition which shifts the terminal state the most. Then we present two PGD-based iterative methods which can be used to solve the resultant constrained maximization problem. Our analysis is not based on Lyapunov theory, and hence requires minimum information of the problem structures. In the model-based setting, we show that the PGD updates can be efficiently performed using back-propagation. In the model-free setting (which is more relevant to ROA analysis of perception-based control), we propose a finite-difference PGD estimate which is general and only requires a black-box simulator for generating the trajectories of the closed-loop system given any initial state. We demonstrate the scalability and generality of our analysis tool on several numerical examples with large-scale NN policies and high-dimensional image observations. We believe that our proposed analysis serves as a meaningful initial step toward further understanding of closed-loop stability of large-scale nonlinear systems and perception-based control.



## **2. Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions**

cs.CL

Survey; 2 figures, 4 tables

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00768v1)

**Authors**: Marwan Omar, Soohyeon Choi, DaeHun Nyang, David Mohaisen

**Abstracts**: Recent natural language processing (NLP) techniques have accomplished high performance on benchmark datasets, primarily due to the significant improvement in the performance of deep learning. The advances in the research community have led to great enhancements in state-of-the-art production systems for NLP tasks, such as virtual assistants, speech recognition, and sentiment analysis. However, such NLP systems still often fail when tested with adversarial attacks. The initial lack of robustness exposed troubling gaps in current models' language understanding capabilities, creating problems when NLP systems are deployed in real life. In this paper, we present a structured overview of NLP robustness research by summarizing the literature in a systemic way across various dimensions. We then take a deep-dive into the various dimensions of robustness, across techniques, metrics, embeddings, and benchmarks. Finally, we argue that robustness should be multi-dimensional, provide insights into current research, identify gaps in the literature to suggest directions worth pursuing to address these gaps.



## **3. DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection**

cs.CR

18 pages, 8 figures; to appear in the Network and Distributed System  Security Symposium (NDSS)

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00763v1)

**Authors**: Phillip Rieger, Thien Duc Nguyen, Markus Miettinen, Ahmad-Reza Sadeghi

**Abstracts**: Federated Learning (FL) allows multiple clients to collaboratively train a Neural Network (NN) model on their private data without revealing the data. Recently, several targeted poisoning attacks against FL have been introduced. These attacks inject a backdoor into the resulting model that allows adversary-controlled inputs to be misclassified. Existing countermeasures against backdoor attacks are inefficient and often merely aim to exclude deviating models from the aggregation. However, this approach also removes benign models of clients with deviating data distributions, causing the aggregated model to perform poorly for such clients.   To address this problem, we propose DeepSight, a novel model filtering approach for mitigating backdoor attacks. It is based on three novel techniques that allow to characterize the distribution of data used to train model updates and seek to measure fine-grained differences in the internal structure and outputs of NNs. Using these techniques, DeepSight can identify suspicious model updates. We also develop a scheme that can accurately cluster model updates. Combining the results of both components, DeepSight is able to identify and eliminate model clusters containing poisoned models with high attack impact. We also show that the backdoor contributions of possibly undetected poisoned models can be effectively mitigated with existing weight clipping-based defenses. We evaluate the performance and effectiveness of DeepSight and show that it can mitigate state-of-the-art backdoor attacks with a negligible impact on the model's performance on benign data.



## **4. Actor-Critic Network for Q&A in an Adversarial Environment**

cs.CL

6 pages, 3 figures, 3 tables

**SubmitDate**: 2022-01-03    [paper-pdf](http://arxiv.org/pdf/2201.00455v1)

**Authors**: Bejan Sadeghian

**Abstracts**: Significant work has been placed in the Q&A NLP space to build models that are more robust to adversarial attacks. Two key areas of focus are in generating adversarial data for the purposes of training against these situations or modifying existing architectures to build robustness within. This paper introduces an approach that joins these two ideas together to train a critic model for use in an almost reinforcement learning framework. Using the Adversarial SQuAD "Add One Sent" dataset we show that there are some promising signs for this method in protecting against Adversarial attacks.



## **5. Towards Transferable Adversarial Attacks on Vision Transformers**

cs.CV

**SubmitDate**: 2022-01-02    [paper-pdf](http://arxiv.org/pdf/2109.04176v3)

**Authors**: Zhipeng Wei, Jingjing Chen, Micah Goldblum, Zuxuan Wu, Tom Goldstein, Yu-Gang Jiang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance on a series of computer vision tasks, yet they still suffer from adversarial examples. % crafted in a similar fashion as CNNs. In this paper, we posit that adversarial attacks on transformers should be specially tailored for their architecture, jointly considering both patches and self-attention, in order to achieve high transferability. More specifically, we introduce a dual attack framework, which contains a Pay No Attention (PNA) attack and a PatchOut attack, to improve the transferability of adversarial samples across different ViTs. We show that skipping the gradients of attention during backpropagation can generate adversarial examples with high transferability. In addition, adversarial perturbations generated by optimizing randomly sampled subsets of patches at each iteration achieve higher attack success rates than attacks using all patches. We evaluate the transferability of attacks on state-of-the-art ViTs, CNNs and robustly trained CNNs. The results of these experiments demonstrate that the proposed dual attack can greatly boost transferability between ViTs and from ViTs to CNNs. In addition, the proposed method can easily be combined with existing transfer methods to boost performance. Code is available at https://github.com/zhipeng-wei/PNA-PatchOut.



## **6. Rethinking Feature Uncertainty in Stochastic Neural Networks for Adversarial Robustness**

cs.LG

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2201.00148v1)

**Authors**: Hao Yang, Min Wang, Zhengfei Yu, Yun Zhou

**Abstracts**: It is well-known that deep neural networks (DNNs) have shown remarkable success in many fields. However, when adding an imperceptible magnitude perturbation on the model input, the model performance might get rapid decrease. To address this issue, a randomness technique has been proposed recently, named Stochastic Neural Networks (SNNs). Specifically, SNNs inject randomness into the model to defend against unseen attacks and improve the adversarial robustness. However, existed studies on SNNs mainly focus on injecting fixed or learnable noises to model weights/activations. In this paper, we find that the existed SNNs performances are largely bottlenecked by the feature representation ability. Surprisingly, simply maximizing the variance per dimension of the feature distribution leads to a considerable boost beyond all previous methods, which we named maximize feature distribution variance stochastic neural network (MFDV-SNN). Extensive experiments on well-known white- and black-box attacks show that MFDV-SNN achieves a significant improvement over existing methods, which indicates that it is a simple but effective method to improve model robustness.



## **7. Characterizing Speech Adversarial Examples Using Self-Attention U-Net Enhancement**

eess.AS

The authors have revised some annotations in Table 4 to improve the  clarity. The authors thank reading feedbacks from Jonathan Le Roux. The first  draft was finished in August 2019. Accepted to IEEE ICASSP 2020

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2003.13917v2)

**Authors**: Chao-Han Huck Yang, Jun Qi, Pin-Yu Chen, Xiaoli Ma, Chin-Hui Lee

**Abstracts**: Recent studies have highlighted adversarial examples as ubiquitous threats to the deep neural network (DNN) based speech recognition systems. In this work, we present a U-Net based attention model, U-Net$_{At}$, to enhance adversarial speech signals. Specifically, we evaluate the model performance by interpretable speech recognition metrics and discuss the model performance by the augmented adversarial training. Our experiments show that our proposed U-Net$_{At}$ improves the perceptual evaluation of speech quality (PESQ) from 1.13 to 2.78, speech transmission index (STI) from 0.65 to 0.75, short-term objective intelligibility (STOI) from 0.83 to 0.96 on the task of speech enhancement with adversarial speech examples. We conduct experiments on the automatic speech recognition (ASR) task with adversarial audio attacks. We find that (i) temporal features learned by the attention network are capable of enhancing the robustness of DNN based ASR models; (ii) the generalization power of DNN based ASR model could be enhanced by applying adversarial training with an additive adversarial data augmentation. The ASR metric on word-error-rates (WERs) shows that there is an absolute 2.22 $\%$ decrease under gradient-based perturbation, and an absolute 2.03 $\%$ decrease, under evolutionary-optimized perturbation, which suggests that our enhancement models with adversarial training can further secure a resilient ASR system.



## **8. An Attention Score Based Attacker for Black-box NLP Classifier**

cs.LG

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2112.11660v2)

**Authors**: Yueyang Liu, Hunmin Lee, Zhipeng Cai

**Abstracts**: Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experiment with our attack model by attacking GRU and 1D-CNN victim models on IMDB datasets. Experimental results demonstrate that our model achieves a higher attack success rate and more efficient than previous methods due to the efficient word selection algorithms are employed and minimized the word substitute number. Also, our model is transferable, which can be used in the image domain with several modifications.



## **9. Adversarial Attack via Dual-Stage Network Erosion**

cs.CV

**SubmitDate**: 2022-01-01    [paper-pdf](http://arxiv.org/pdf/2201.00097v1)

**Authors**: Yexin Duan, Junhua Zou, Xingyu Zhou, Wu Zhang, Jin Zhang, Zhisong Pan

**Abstracts**: Deep neural networks are vulnerable to adversarial examples, which can fool deep models by adding subtle perturbations. Although existing attacks have achieved promising results, it still leaves a long way to go for generating transferable adversarial examples under the black-box setting. To this end, this paper proposes to improve the transferability of adversarial examples, and applies dual-stage feature-level perturbations to an existing model to implicitly create a set of diverse models. Then these models are fused by the longitudinal ensemble during the iterations. The proposed method is termed Dual-Stage Network Erosion (DSNE). We conduct comprehensive experiments both on non-residual and residual networks, and obtain more transferable adversarial examples with the computational cost similar to the state-of-the-art method. In particular, for the residual networks, the transferability of the adversarial examples can be significantly improved by biasing the residual block information to the skip connections. Our work provides new insights into the architectural vulnerability of neural networks and presents new challenges to the robustness of neural networks.



## **10. Privacy-Protecting COVID-19 Exposure Notification Based on Cluster Events**

cs.CR

11 pages. This paper was presented at the NIST Workshop on Challenges  for Digital Proximity Detection in Pandemics: Privacy, Accuracy, and Impact,  January 28 02021

**SubmitDate**: 2021-12-31    [paper-pdf](http://arxiv.org/pdf/2201.00031v1)

**Authors**: Paul Syverson

**Abstracts**: We provide a rough sketch of a simple system design for exposure notification of COVID-19 infections based on copresence at cluster events -- locations and times where a threshold number of tested-positive (TP) individuals were present. Unlike other designs, such as DP3T or the Apple-Google exposure-notification system, this design does not track or notify based on detecting direct proximity to TP individuals.   The design makes use of existing or in-development tests for COVID-19 that are relatively cheap and return results in less than an hour, and that have high specificity but may have lower sensitivity. It also uses readily available location tracking for mobile phones and similar devices. It reports events at which TP individuals were present but does not link events with individuals or with other events in an individual's history. Participating individuals are notified of detected cluster events. They can then compare these locally to their own location history. Detected cluster events can be publicized through public channels. Thus, individuals not participating in the reporting system can still be notified of exposure.   A proper security analysis is beyond the scope of this design sketch. We do, however, discuss resistance to various adversaries and attacks on privacy as well as false-reporting attacks.



## **11. QueryNet: Attack by Multi-Identity Surrogates**

cs.LG

QueryNet reduces queries by about an order of magnitude against SOTA  black-box attacks

**SubmitDate**: 2021-12-31    [paper-pdf](http://arxiv.org/pdf/2105.15010v3)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstracts**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/AllenChen1998/QueryNet.



## **12. NCIS: Neural Contextual Iterative Smoothing for Purifying Adversarial Perturbations**

cs.CV

Preprint version

**SubmitDate**: 2021-12-30    [paper-pdf](http://arxiv.org/pdf/2106.11644v2)

**Authors**: Sungmin Cha, Naeun Ko, Youngjoon Yoo, Taesup Moon

**Abstracts**: We propose a novel and effective purification based adversarial defense method against pre-processor blind white- and black-box attacks. Our method is computationally efficient and trained only with self-supervised learning on general images, without requiring any adversarial training or retraining of the classification model. We first show an empirical analysis on the adversarial noise, defined to be the residual between an original image and its adversarial example, has almost zero mean, symmetric distribution. Based on this observation, we propose a very simple iterative Gaussian Smoothing (GS) which can effectively smooth out adversarial noise and achieve substantially high robust accuracy. To further improve it, we propose Neural Contextual Iterative Smoothing (NCIS), which trains a blind-spot network (BSN) in a self-supervised manner to reconstruct the discriminative features of the original image that is also smoothed out by GS. From our extensive experiments on the large-scale ImageNet using four classification models, we show that our method achieves both competitive standard accuracy and state-of-the-art robust accuracy against most strong purifier-blind white- and black-box attacks. Also, we propose a new benchmark for evaluating a purification method based on commercial image classification APIs, such as AWS, Azure, Clarifai and Google. We generate adversarial examples by ensemble transfer-based black-box attack, which can induce complete misclassification of APIs, and demonstrate that our method can be used to increase adversarial robustness of APIs.



## **13. Efficient Robust Training via Backward Smoothing**

cs.LG

12 pages, 15 tables, 6 figures. In AAAI 2022

**SubmitDate**: 2021-12-30    [paper-pdf](http://arxiv.org/pdf/2010.01278v2)

**Authors**: Jinghui Chen, Yu Cheng, Zhe Gan, Quanquan Gu, Jingjing Liu

**Abstracts**: Adversarial training is so far the most effective strategy in defending against adversarial examples. However, it suffers from high computational costs due to the iterative adversarial attacks in each training step. Recent studies show that it is possible to achieve fast Adversarial Training by performing a single-step attack with random initialization. However, such an approach still lags behind state-of-the-art adversarial training algorithms on both stability and model robustness. In this work, we develop a new understanding towards Fast Adversarial Training, by viewing random initialization as performing randomized smoothing for better optimization of the inner maximization problem. Following this new perspective, we also propose a new initialization strategy, backward smoothing, to further improve the stability and model robustness over single-step robust training methods. Experiments on multiple benchmarks demonstrate that our method achieves similar model robustness as the original TRADES method while using much less training time ($\sim$3x improvement with the same training schedule).



## **14. Improved Gradient based Adversarial Attacks for Quantized Networks**

cs.CV

AAAI 2022

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2003.13511v2)

**Authors**: Kartik Gupta, Thalaiyasingam Ajanthan

**Abstracts**: Neural network quantization has become increasingly popular due to efficient memory consumption and faster computation resulting from bitwise operations on the quantized networks. Even though they exhibit excellent generalization capabilities, their robustness properties are not well-understood. In this work, we systematically study the robustness of quantized networks against gradient based adversarial attacks and demonstrate that these quantized models suffer from gradient vanishing issues and show a fake sense of robustness. By attributing gradient vanishing to poor forward-backward signal propagation in the trained network, we introduce a simple temperature scaling approach to mitigate this issue while preserving the decision boundary. Despite being a simple modification to existing gradient based adversarial attacks, experiments on multiple image classification datasets with multiple network architectures demonstrate that our temperature scaled attacks obtain near-perfect success rate on quantized networks while outperforming original attacks on adversarially trained models as well as floating-point networks. Code is available at https://github.com/kartikgupta-at-anu/attack-bnn.



## **15. Perfectly Secure Message Transmission against Rational Adversaries**

cs.CR

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2009.07513v3)

**Authors**: Maiki Fujita, Takeshi Koshiba, Kenji Yasunaga

**Abstracts**: Secure Message Transmission (SMT) is a two-party cryptographic protocol by which the sender can securely and reliably transmit messages to the receiver using multiple channels. An adversary can corrupt a subset of the channels and commit eavesdropping and tampering attacks over the channels. In this work, we introduce a game-theoretic security model for SMT in which adversaries have some preferences for protocol execution. We define rational "timid" adversaries who prefer to violate security requirements but do not prefer the tampering to be detected.   First, we consider the basic setting where a single adversary attacks the protocol. We construct perfect SMT protocols against any rational adversary corrupting all but one of the channels. Since minority corruption is required in the traditional setting, our results demonstrate a way of circumventing the cryptographic impossibility results by a game-theoretic approach.   Next, we study the setting in which all the channels can be corrupted by multiple adversaries who do not cooperate. Since we cannot hope for any security if a single adversary corrupts all the channels or multiple adversaries cooperate maliciously, the scenario can arise from a game-theoretic model. We also study the scenario in which both malicious and rational adversaries exist.



## **16. Domain Knowledge Alleviates Adversarial Attacks in Multi-Label Classifiers**

cs.LG

Accepted for publications in IEEE TPAMI journal

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2006.03833v4)

**Authors**: Stefano Melacci, Gabriele Ciravegna, Angelo Sotgiu, Ambra Demontis, Battista Biggio, Marco Gori, Fabio Roli

**Abstracts**: Adversarial attacks on machine learning-based classifiers, along with defense mechanisms, have been widely studied in the context of single-label classification problems. In this paper, we shift the attention to multi-label classification, where the availability of domain knowledge on the relationships among the considered classes may offer a natural way to spot incoherent predictions, i.e., predictions associated to adversarial examples lying outside of the training data distribution. We explore this intuition in a framework in which first-order logic knowledge is converted into constraints and injected into a semi-supervised learning problem. Within this setting, the constrained classifier learns to fulfill the domain knowledge over the marginal distribution, and can naturally reject samples with incoherent predictions. Even though our method does not exploit any knowledge of attacks during training, our experimental analysis surprisingly unveils that domain-knowledge constraints can help detect adversarial examples effectively, especially if such constraints are not known to the attacker.



## **17. Invertible Image Dataset Protection**

cs.CV

Submitted to ICME 2022. Authors are from University of Science and  Technology of China, Fudan University, China. A potential extended version of  this work is under way

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2112.14420v1)

**Authors**: Kejiang Chen, Xianhan Zeng, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Deep learning has achieved enormous success in various industrial applications. Companies do not want their valuable data to be stolen by malicious employees to train pirated models. Nor do they wish the data analyzed by the competitors after using them online. We propose a novel solution for dataset protection in this scenario by robustly and reversibly transform the images into adversarial images. We develop a reversible adversarial example generator (RAEG) that introduces slight changes to the images to fool traditional classification models. Even though malicious attacks train pirated models based on the defensed versions of the protected images, RAEG can significantly weaken the functionality of these models. Meanwhile, the reversibility of RAEG ensures the performance of authorized models. Extensive experiments demonstrate that RAEG can better protect the data with slight distortion against adversarial defense than previous methods.



## **18. Super-Efficient Super Resolution for Fast Adversarial Defense at the Edge**

eess.IV

This preprint is for personal use only. The official article will  appear in proceedings of Design, Automation & Test in Europe (DATE), 2022, as  part of the Special Initiative on Autonomous Systems Design (ASD)

**SubmitDate**: 2021-12-29    [paper-pdf](http://arxiv.org/pdf/2112.14340v1)

**Authors**: Kartikeya Bhardwaj, Dibakar Gope, James Ward, Paul Whatmough, Danny Loh

**Abstracts**: Autonomous systems are highly vulnerable to a variety of adversarial attacks on Deep Neural Networks (DNNs). Training-free model-agnostic defenses have recently gained popularity due to their speed, ease of deployment, and ability to work across many DNNs. To this end, a new technique has emerged for mitigating attacks on image classification DNNs, namely, preprocessing adversarial images using super resolution -- upscaling low-quality inputs into high-resolution images. This defense requires running both image classifiers and super resolution models on constrained autonomous systems. However, super resolution incurs a heavy computational cost. Therefore, in this paper, we investigate the following question: Does the robustness of image classifiers suffer if we use tiny super resolution models? To answer this, we first review a recent work called Super-Efficient Super Resolution (SESR) that achieves similar or better image quality than prior art while requiring 2x to 330x fewer Multiply-Accumulate (MAC) operations. We demonstrate that despite being orders of magnitude smaller than existing models, SESR achieves the same level of robustness as significantly larger networks. Finally, we estimate end-to-end performance of super resolution-based defenses on a commercial Arm Ethos-U55 micro-NPU. Our findings show that SESR achieves nearly 3x higher FPS than a baseline while achieving similar robustness.



## **19. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

cs.LG

19 pages, 7 figures, 5 tables, submitted to Astronomy & Computing

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.14299v1)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: Data processing and analysis pipelines in cosmological survey experiments introduce data perturbations that can significantly degrade the performance of deep learning-based models. Given the increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.



## **20. Constrained Gradient Descent: A Powerful and Principled Evasion Attack Against Neural Networks**

cs.LG

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.14232v1)

**Authors**: Weiran Lin, Keane Lucas, Lujo Bauer, Michael K. Reiter, Mahmood Sharif

**Abstracts**: Minimal adversarial perturbations added to inputs have been shown to be effective at fooling deep neural networks. In this paper, we introduce several innovations that make white-box targeted attacks follow the intuition of the attacker's goal: to trick the model to assign a higher probability to the target class than to any other, while staying within a specified distance from the original input. First, we propose a new loss function that explicitly captures the goal of targeted attacks, in particular, by using the logits of all classes instead of just a subset, as is common. We show that Auto-PGD with this loss function finds more adversarial examples than it does with other commonly used loss functions. Second, we propose a new attack method that uses a further developed version of our loss function capturing both the misclassification objective and the $L_{\infty}$ distance limit $\epsilon$. This new attack method is relatively 1.5--4.2% more successful on the CIFAR10 dataset and relatively 8.2--14.9% more successful on the ImageNet dataset, than the next best state-of-the-art attack. We confirm using statistical tests that our attack outperforms state-of-the-art attacks on different datasets and values of $\epsilon$ and against different defenses.



## **21. Understanding and Measuring Robustness of Multimodal Learning**

cs.LG

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.12792v2)

**Authors**: Nishant Vishwamitra, Hongxin Hu, Ziming Zhao, Long Cheng, Feng Luo

**Abstracts**: The modern digital world is increasingly becoming multimodal. Although multimodal learning has recently revolutionized the state-of-the-art performance in multimodal tasks, relatively little is known about the robustness of multimodal learning in an adversarial setting. In this paper, we introduce a comprehensive measurement of the adversarial robustness of multimodal learning by focusing on the fusion of input modalities in multimodal models, via a framework called MUROAN (MUltimodal RObustness ANalyzer). We first present a unified view of multimodal models in MUROAN and identify the fusion mechanism of multimodal models as a key vulnerability. We then introduce a new type of multimodal adversarial attacks called decoupling attack in MUROAN that aims to compromise multimodal models by decoupling their fused modalities. We leverage the decoupling attack of MUROAN to measure several state-of-the-art multimodal models and find that the multimodal fusion mechanism in all these models is vulnerable to decoupling attacks. We especially demonstrate that, in the worst case, the decoupling attack of MUROAN achieves an attack success rate of 100% by decoupling just 1.16% of the input space. Finally, we show that traditional adversarial training is insufficient to improve the robustness of multimodal models with respect to decoupling attacks. We hope our findings encourage researchers to pursue improving the robustness of multimodal learning.



## **22. Mind Your Solver! On Adversarial Attack and Defense for Combinatorial Optimization**

math.OC

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2201.00402v1)

**Authors**: Han Lu, Zenan Li, Runzhong Wang, Qibing Ren, Junchi Yan, Xiaokang Yang

**Abstracts**: Combinatorial optimization (CO) is a long-standing challenging task not only in its inherent complexity (e.g. NP-hard) but also the possible sensitivity to input conditions. In this paper, we take an initiative on developing the mechanisms for adversarial attack and defense towards combinatorial optimization solvers, whereby the solver is treated as a black-box function and the original problem's underlying graph structure (which is often available and associated with the problem instance, e.g. DAG, TSP) is attacked under a given budget. In particular, we present a simple yet effective defense strategy to modify the graph structure to increase the robustness of solvers, which shows its universal effectiveness across tasks and solvers.



## **23. Boosting the Transferability of Video Adversarial Examples via Temporal Translation**

cs.CV

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2110.09075v2)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Although deep-learning based video recognition models have achieved remarkable success, they are vulnerable to adversarial examples that are generated by adding human-imperceptible perturbations on clean video samples. As indicated in recent studies, adversarial examples are transferable, which makes it feasible for black-box attacks in real-world applications. Nevertheless, most existing adversarial attack methods have poor transferability when attacking other video models and transfer-based attacks on video models are still unexplored. To this end, we propose to boost the transferability of video adversarial examples for black-box attacks on video recognition models. Through extensive analysis, we discover that different video recognition models rely on different discriminative temporal patterns, leading to the poor transferability of video adversarial examples. This motivates us to introduce a temporal translation attack method, which optimizes the adversarial perturbations over a set of temporal translated video clips. By generating adversarial examples over translated videos, the resulting adversarial examples are less sensitive to temporal patterns existed in the white-box model being attacked and thus can be better transferred. Extensive experiments on the Kinetics-400 dataset and the UCF-101 dataset demonstrate that our method can significantly boost the transferability of video adversarial examples. For transfer-based attack against video recognition models, it achieves a 61.56% average attack success rate on the Kinetics-400 and 48.60% on the UCF-101. Code is available at https://github.com/zhipeng-wei/TT.



## **24. On anti-stochastic properties of unlabeled graphs**

cs.DM

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.04395v2)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstracts**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.



## **25. Associative Adversarial Learning Based on Selective Attack**

cs.CV

**SubmitDate**: 2021-12-28    [paper-pdf](http://arxiv.org/pdf/2112.13989v1)

**Authors**: Runqi Wang, Xiaoyue Duan, Baochang Zhang, Song Xue, Wentao Zhu, David Doermann, Guodong Guo

**Abstracts**: A human's attention can intuitively adapt to corrupted areas of an image by recalling a similar uncorrupted image they have previously seen. This observation motivates us to improve the attention of adversarial images by considering their clean counterparts. To accomplish this, we introduce Associative Adversarial Learning (AAL) into adversarial learning to guide a selective attack. We formulate the intrinsic relationship between attention and attack (perturbation) as a coupling optimization problem to improve their interaction. This leads to an attention backtracking algorithm that can effectively enhance the attention's adversarial robustness. Our method is generic and can be used to address a variety of tasks by simply choosing different kernels for the associative attention that select other regions for a specific attack. Experimental results show that the selective attack improves the model's performance. We show that our method improves the recognition accuracy of adversarial training on ImageNet by 8.32% compared with the baseline. It also increases object detection mAP on PascalVOC by 2.02% and recognition accuracy of few-shot learning on miniImageNet by 1.63%.



## **26. PORTFILER: Port-Level Network Profiling for Self-Propagating Malware Detection**

cs.CR

An earlier version is accepted to be published in IEEE Conference on  Communications and Network Security (CNS) 2021

**SubmitDate**: 2021-12-27    [paper-pdf](http://arxiv.org/pdf/2112.13798v1)

**Authors**: Talha Ongun, Oliver Spohngellert, Benjamin Miller, Simona Boboila, Alina Oprea, Tina Eliassi-Rad, Jason Hiser, Alastair Nottingham, Jack Davidson, Malathi Veeraraghavan

**Abstracts**: Recent self-propagating malware (SPM) campaigns compromised hundred of thousands of victim machines on the Internet. It is challenging to detect these attacks in their early stages, as adversaries utilize common network services, use novel techniques, and can evade existing detection mechanisms. We propose PORTFILER (PORT-Level Network Traffic ProFILER), a new machine learning system applied to network traffic for detecting SPM attacks. PORTFILER extracts port-level features from the Zeek connection logs collected at a border of a monitored network, applies anomaly detection techniques to identify suspicious events, and ranks the alerts across ports for investigation by the Security Operations Center (SOC). We propose a novel ensemble methodology for aggregating individual models in PORTFILER that increases resilience against several evasion strategies compared to standard ML baselines. We extensively evaluate PORTFILER on traffic collected from two university networks, and show that it can detect SPM attacks with different patterns, such as WannaCry and Mirai, and performs well under evasion. Ranking across ports achieves precision over 0.94 with low false positive rates in the top ranked alerts. When deployed on the university networks, PORTFILER detected anomalous SPM-like activity on one of the campus networks, confirmed by the university SOC as malicious. PORTFILER also detected a Mirai attack recreated on the two university networks with higher precision and recall than deep-learning-based autoencoder methods.



## **27. Adversarial Attack for Asynchronous Event-based Data**

cs.CV

8 pages, 6 figures, Thirty-Sixth AAAI Conference on Artificial  Intelligence (AAAI-22)

**SubmitDate**: 2021-12-27    [paper-pdf](http://arxiv.org/pdf/2112.13534v1)

**Authors**: Wooju Lee, Hyun Myung

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial examples that are carefully designed to cause the deep learning model to make mistakes. Adversarial examples of 2D images and 3D point clouds have been extensively studied, but studies on event-based data are limited. Event-based data can be an alternative to a 2D image under high-speed movements, such as autonomous driving. However, the given adversarial events make the current deep learning model vulnerable to safety issues. In this work, we generate adversarial examples and then train the robust models for event-based data, for the first time. Our algorithm shifts the time of the original events and generates additional adversarial events. Additional adversarial events are generated in two stages. First, null events are added to the event-based data to generate additional adversarial events. The perturbation size can be controlled with the number of null events. Second, the location and time of additional adversarial events are set to mislead DNNs in a gradient-based attack. Our algorithm achieves an attack success rate of 97.95\% on the N-Caltech101 dataset. Furthermore, the adversarial training model improves robustness on the adversarial event data compared to the original model.



## **28. Killing One Bird with Two Stones: Model Extraction and Attribute Inference Attacks against BERT-based APIs**

cs.CR

**SubmitDate**: 2021-12-26    [paper-pdf](http://arxiv.org/pdf/2105.10909v2)

**Authors**: Chen Chen, Xuanli He, Lingjuan Lyu, Fangzhao Wu

**Abstracts**: The collection and availability of big data, combined with advances in pre-trained models (e.g., BERT, XLNET, etc), have revolutionized the predictive performance of modern natural language processing tasks, ranging from text classification to text generation. This allows corporations to provide machine learning as a service (MLaaS) by encapsulating fine-tuned BERT-based models as APIs. However, BERT-based APIs have exhibited a series of security and privacy vulnerabilities. For example, prior work has exploited the security issues of the BERT-based APIs through the adversarial examples crafted by the extracted model. However, the privacy leakage problems of the BERT-based APIs through the extracted model have not been well studied. On the other hand, due to the high capacity of BERT-based APIs, the fine-tuned model is easy to be overlearned, but what kind of information can be leaked from the extracted model remains unknown. In this work, we bridge this gap by first presenting an effective model extraction attack, where the adversary can practically steal a BERT-based API (the target/victim model) by only querying a limited number of queries. We further develop an effective attribute inference attack which can infer the sensitive attribute of the training data used by the BERT-based APIs. Our extensive experiments on benchmark datasets under various realistic settings validate the potential vulnerabilities of BERT-based APIs. Moreover, we demonstrate that two promising defense methods become ineffective against our attacks, which calls for more effective defense methods.



## **29. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

cs.LG

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2112.13267v1)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Sayan Ranu, Arnab Bhattacharya

**Abstracts**: Graph neural networks (GNNs) have witnessed significant adoption in the industry owing to impressive performance on various predictive tasks. Performance alone, however, is not enough. Any widely deployed machine learning algorithm must be robust to adversarial attacks. In this work, we investigate this aspect for GNNs, identify vulnerabilities, and link them to graph properties that may potentially lead to the development of more secure and robust GNNs. Specifically, we formulate the problem of task and model agnostic evasion attacks where adversaries modify the test graph to affect the performance of any unknown downstream task. The proposed algorithm, GRAND ($Gr$aph $A$ttack via $N$eighborhood $D$istortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, GRAND designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep $Q$-learning. Extensive experiments on real datasets show that, on average, GRAND is up to $50\%$ more effective than state of the art techniques, while being more than $100$ times faster.



## **30. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2111.10844v3)

**Authors**: Kaiyuan Liu, Xingyu Li, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo, Jisong Guan, Yi Zhou

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.



## **31. Stealthy Attack on Algorithmic-Protected DNNs via Smart Bit Flipping**

cs.CR

Accepted for the 23rd International Symposium on Quality Electronic  Design (ISQED'22)

**SubmitDate**: 2021-12-25    [paper-pdf](http://arxiv.org/pdf/2112.13162v1)

**Authors**: Behnam Ghavami, Seyd Movi, Zhenman Fang, Lesley Shannon

**Abstracts**: Recently, deep neural networks (DNNs) have been deployed in safety-critical systems such as autonomous vehicles and medical devices. Shortly after that, the vulnerability of DNNs were revealed by stealthy adversarial examples where crafted inputs -- by adding tiny perturbations to original inputs -- can lead a DNN to generate misclassification outputs. To improve the robustness of DNNs, some algorithmic-based countermeasures against adversarial examples have been introduced thereafter.   In this paper, we propose a new type of stealthy attack on protected DNNs to circumvent the algorithmic defenses: via smart bit flipping in DNN weights, we can reserve the classification accuracy for clean inputs but misclassify crafted inputs even with algorithmic countermeasures. To fool protected DNNs in a stealthy way, we introduce a novel method to efficiently find their most vulnerable weights and flip those bits in hardware. Experimental results show that we can successfully apply our stealthy attack against state-of-the-art algorithmic-protected DNNs.



## **32. SoK: A Study of the Security on Voice Processing Systems**

cs.CR

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13144v1)

**Authors**: Robert Chang, Logan Kuo, Arthur Liu, Nader Sehatbakhsh

**Abstracts**: As the use of Voice Processing Systems (VPS) continues to become more prevalent in our daily lives through the increased reliance on applications such as commercial voice recognition devices as well as major text-to-speech software, the attacks on these systems are increasingly complex, varied, and constantly evolving. With the use cases for VPS rapidly growing into new spaces and purposes, the potential consequences regarding privacy are increasingly more dangerous. In addition, the growing number and increased practicality of over-the-air attacks have made system failures much more probable. In this paper, we will identify and classify an arrangement of unique attacks on voice processing systems. Over the years research has been moving from specialized, untargeted attacks that result in the malfunction of systems and the denial of services to more general, targeted attacks that can force an outcome controlled by an adversary. The current and most frequently used machine learning systems and deep neural networks, which are at the core of modern voice processing systems, were built with a focus on performance and scalability rather than security. Therefore, it is critical for us to reassess the developing voice processing landscape and to identify the state of current attacks and defenses so that we may suggest future developments and theoretical improvements.



## **33. CatchBackdoor: Backdoor Testing by Critical Trojan Neural Path Identification via Differential Fuzzing**

cs.CR

13 pages

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13064v1)

**Authors**: Haibo Jin, Ruoxi Chen, Jinyin Chen, Yao Cheng, Chong Fu, Ting Wang, Yue Yu, Zhaoyan Ming

**Abstracts**: The success of deep neural networks (DNNs) in real-world applications has benefited from abundant pre-trained models. However, the backdoored pre-trained models can pose a significant trojan threat to the deployment of downstream DNNs. Existing DNN testing methods are mainly designed to find incorrect corner case behaviors in adversarial settings but fail to discover the backdoors crafted by strong trojan attacks. Observing the trojan network behaviors shows that they are not just reflected by a single compromised neuron as proposed by previous work but attributed to the critical neural paths in the activation intensity and frequency of multiple neurons. This work formulates the DNN backdoor testing and proposes the CatchBackdoor framework. Via differential fuzzing of critical neurons from a small number of benign examples, we identify the trojan paths and particularly the critical ones, and generate backdoor testing examples by simulating the critical neurons in the identified paths. Extensive experiments demonstrate the superiority of CatchBackdoor, with higher detection performance than existing methods. CatchBackdoor works better on detecting backdoors by stealthy blending and adaptive attacks, which existing methods fail to detect. Moreover, our experiments show that CatchBackdoor may reveal the potential backdoors of models in Model Zoo.



## **34. NIP: Neuron-level Inverse Perturbation Against Adversarial Attacks**

cs.CV

14 pages

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13060v1)

**Authors**: Ruoxi Chen, Haibo Jin, Jinyin Chen, Haibin Zheng, Yue Yu, Shouling Ji

**Abstracts**: Although deep learning models have achieved unprecedented success, their vulnerabilities towards adversarial attacks have attracted increasing attention, especially when deployed in security-critical domains. To address the challenge, numerous defense strategies, including reactive and proactive ones, have been proposed for robustness improvement. From the perspective of image feature space, some of them cannot reach satisfying results due to the shift of features. Besides, features learned by models are not directly related to classification results. Different from them, We consider defense method essentially from model inside and investigated the neuron behaviors before and after attacks. We observed that attacks mislead the model by dramatically changing the neurons that contribute most and least to the correct label. Motivated by it, we introduce the concept of neuron influence and further divide neurons into front, middle and tail part. Based on it, we propose neuron-level inverse perturbation(NIP), the first neuron-level reactive defense method against adversarial attacks. By strengthening front neurons and weakening those in the tail part, NIP can eliminate nearly all adversarial perturbations while still maintaining high benign accuracy. Besides, it can cope with different sizes of perturbations via adaptivity, especially larger ones. Comprehensive experiments conducted on three datasets and six models show that NIP outperforms the state-of-the-art baselines against eleven adversarial attacks. We further provide interpretable proofs via neuron activation and visualization for better understanding.



## **35. One Bad Apple Spoils the Bunch: Transaction DoS in MimbleWimble Blockchains**

cs.CR

9 pages, 4 figures

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.13009v1)

**Authors**: Seyed Ali Tabatabaee, Charlene Nicer, Ivan Beschastnikh, Chen Feng

**Abstracts**: As adoption of blockchain-based systems grows, more attention is being given to privacy of these systems. Early systems like BitCoin provided few privacy features. As a result, systems with strong privacy guarantees, including Monero, Zcash, and MimbleWimble have been developed. Compared to BitCoin, these cryptocurrencies are much less understood. In this paper, we focus on MimbleWimble, which uses the Dandelion++ protocol for private transaction relay and transaction aggregation to provide transaction content privacy. We find that in combination these two features make MimbleWimble susceptible to a new type of denial-of-service attacks. We design, prototype, and evaluate this attack on the Beam network using a private test network and a network simulator. We find that by controlling only 10% of the network nodes, the adversary can prevent over 45% of all transactions from ending up in the blockchain. We also discuss several potential approaches for mitigating this attack.



## **36. Parameter identifiability of a deep feedforward ReLU neural network**

math.ST

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12982v1)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstracts**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.



## **37. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

cs.LG

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12376v2)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) has become a widely recognized defense mechanism to improve the robustness of deep neural networks against adversarial attacks. It solves a min-max optimization problem, where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the min-max nature makes AT computationally intensive and thus difficult to scale. Meanwhile, the FAST-AT algorithm, and in fact many recent algorithms that improve AT, simplify the min-max based AT by replacing its maximization step with the simple one-shot gradient sign based attack generation step. Although easy to implement, FAST-AT lacks theoretical guarantees, and its practical performance can be unsatisfactory, suffering from the robustness catastrophic overfitting when training with strong adversaries.   In this paper, we propose to design FAST-AT from the perspective of bi-level optimization (BLO). We first make the key observation that the most commonly-used algorithmic specification of FAST-AT is equivalent to using some gradient descent-type algorithm to solve a bi-level problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Based on the above observation, we propose a new tractable bi-level optimization problem, design and analyze a new set of algorithms termed Fast Bi-level AT (FAST-BAT). FAST-BAT is capable of defending sign-based projected gradient descent (PGD) attacks without calling any gradient sign method and explicit robust regularization. Furthermore, we empirically show that our method outperforms state-of-the-art FAST-AT baselines, by achieving superior model robustness without inducing robustness catastrophic overfitting, or suffering from any loss of standard accuracy.



## **38. Robust Secretary and Prophet Algorithms for Packing Integer Programs**

cs.DS

Appears in SODA 2022

**SubmitDate**: 2021-12-24    [paper-pdf](http://arxiv.org/pdf/2112.12920v1)

**Authors**: C. J. Argue, Anupam Gupta, Marco Molinaro, Sahil Singla

**Abstracts**: We study the problem of solving Packing Integer Programs (PIPs) in the online setting, where columns in $[0,1]^d$ of the constraint matrix are revealed sequentially, and the goal is to pick a subset of the columns that sum to at most $B$ in each coordinate while maximizing the objective. Excellent results are known in the secretary setting, where the columns are adversarially chosen, but presented in a uniformly random order. However, these existing algorithms are susceptible to adversarial attacks: they try to "learn" characteristics of a good solution, but tend to over-fit to the model, and hence a small number of adversarial corruptions can cause the algorithm to fail.   In this paper, we give the first robust algorithms for Packing Integer Programs, specifically in the recently proposed Byzantine Secretary framework. Our techniques are based on a two-level use of online learning, to robustly learn an approximation to the optimal value, and then to use this robust estimate to pick a good solution. These techniques are general and we use them to design robust algorithms for PIPs in the prophet model as well, specifically in the Prophet-with-Augmentations framework. We also improve known results in the Byzantine Secretary framework: we make the non-constructive results algorithmic and improve the existing bounds for single-item and matroid constraints.



## **39. Adaptive Modeling Against Adversarial Attacks**

cs.LG

10 pages, 3 figures

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12431v1)

**Authors**: Zhiwen Yan, Teck Khim Ng

**Abstracts**: Adversarial training, the process of training a deep learning model with adversarial data, is one of the most successful adversarial defense methods for deep learning models. We have found that the robustness to white-box attack of an adversarially trained model can be further improved if we fine tune this model in inference stage to adapt to the adversarial input, with the extra information in it. We introduce an algorithm that "post trains" the model at inference stage between the original output class and a "neighbor" class, with existing training data. The accuracy of pre-trained Fast-FGSM CIFAR10 classifier base model against white-box projected gradient attack (PGD) can be significantly improved from 46.8% to 64.5% with our algorithm.



## **40. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

cs.CR

**SubmitDate**: 2021-12-23    [paper-pdf](http://arxiv.org/pdf/2112.12310v1)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstracts**: The malware has been being one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against the ever-increasing and ever-evolving threats of malware, tremendous efforts have been made to propose a variety of malware detection methods that attempt to effectively and efficiently detect malware. Recent studies have shown that, on the one hand, existing ML and DL enable the superior detection of newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples, which are maliciously generated by slightly and carefully perturbing the legitimate inputs to confuse the targeted models. Basically, adversarial attacks are initially extensively studied in the domain of computer vision, and some quickly expanded to other domains, including NLP, speech recognition and even malware detection. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of PE malware. We then conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of PE malware detection. We conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities.



## **41. Detect & Reject for Transferability of Black-box Adversarial Attacks Against Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12095v1)

**Authors**: Islam Debicha, Thibault Debatty, Jean-Michel Dricot, Wim Mees, Tayeb Kenaza

**Abstracts**: In the last decade, the use of Machine Learning techniques in anomaly-based intrusion detection systems has seen much success. However, recent studies have shown that Machine learning in general and deep learning specifically are vulnerable to adversarial attacks where the attacker attempts to fool models by supplying deceptive input. Research in computer vision, where this vulnerability was first discovered, has shown that adversarial images designed to fool a specific model can deceive other machine learning models. In this paper, we investigate the transferability of adversarial network traffic against multiple machine learning-based intrusion detection systems. Furthermore, we analyze the robustness of the ensemble intrusion detection system, which is notorious for its better accuracy compared to a single model, against the transferability of adversarial attacks. Finally, we examine Detect & Reject as a defensive mechanism to limit the effect of the transferability property of adversarial network traffic against machine learning-based intrusion detection systems.



## **42. Evaluating the Robustness of Deep Reinforcement Learning for Autonomous and Adversarial Policies in a Multi-agent Urban Driving Environment**

cs.AI

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11947v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is actively used for training autonomous driving agents in a vision-based urban simulated environment. Due to the large availability of various reinforcement learning algorithms, we are still unsure of which one works better while training autonomous cars in single-agent as well as multi-agent driving environments. A comparison of deep reinforcement learning in vision-based autonomous driving will open up the possibilities for training better autonomous car policies. Also, autonomous cars trained on deep reinforcement learning-based algorithms are known for being vulnerable to adversarial attacks, and we have less information on which algorithms would act as a good adversarial agent. In this work, we provide a systematic evaluation and comparative analysis of 6 deep reinforcement learning algorithms for autonomous and adversarial driving in four-way intersection scenario. Specifically, we first train autonomous cars using state-of-the-art deep reinforcement learning algorithms. Second, we test driving capabilities of the trained autonomous policies in single-agent as well as multi-agent scenarios. Lastly, we use the same deep reinforcement learning algorithms to train adversarial driving agents, in order to test the driving performance of autonomous cars and look for possible collision and offroad driving scenarios. We perform experiments by using vision-only high fidelity urban driving simulated environments.



## **43. Adversarial Deep Reinforcement Learning for Trustworthy Autonomous Driving Policies**

cs.AI

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11937v1)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is widely used to train autonomous cars in a simulated environment. Still, autonomous cars are well known for being vulnerable when exposed to adversarial attacks. This raises the question of whether we can train the adversary as a driving agent for finding failure scenarios in autonomous cars, and then retrain autonomous cars with new adversarial inputs to improve their robustness. In this work, we first train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars in a multi-agent setting. Second, we verify that adversarial examples can be used not only for finding unwanted autonomous driving behavior, but also for helping autonomous driving cars in improving their deep reinforcement learning policies. By using a high fidelity urban driving simulation environment and vision-based driving agents, we demonstrate that the autonomous cars retrained using the adversary player noticeably increase the performance of their driving policies in terms of reducing collision and offroad steering errors.



## **44. Consistency Regularization for Adversarial Robustness**

cs.LG

Published as a conference proceeding for AAAI 2022

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2103.04623v3)

**Authors**: Jihoon Tack, Sihyun Yu, Jongheon Jeong, Minseon Kim, Sung Ju Hwang, Jinwoo Shin

**Abstracts**: Adversarial training (AT) is currently one of the most successful methods to obtain the adversarial robustness of deep neural networks. However, the phenomenon of robust overfitting, i.e., the robustness starts to decrease significantly during AT, has been problematic, not only making practitioners consider a bag of tricks for a successful training, e.g., early stopping, but also incurring a significant generalization gap in the robustness. In this paper, we propose an effective regularization technique that prevents robust overfitting by optimizing an auxiliary `consistency' regularization loss during AT. Specifically, we discover that data augmentation is a quite effective tool to mitigate the overfitting in AT, and develop a regularization that forces the predictive distributions after attacking from two different augmentations of the same instance to be similar with each other. Our experimental results demonstrate that such a simple regularization technique brings significant improvements in the test robust accuracy of a wide range of AT methods. More remarkably, we also show that our method could significantly help the model to generalize its robustness against unseen adversaries, e.g., other types or larger perturbations compared to those used during training. Code is available at https://github.com/alinlab/consistency-adversarial.



## **45. How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?**

cs.CL

Accepted by NeurIPS-2021

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.11668v1)

**Authors**: Xinhsuai Dong, Luu Anh Tuan, Min Lin, Shuicheng Yan, Hanwang Zhang

**Abstracts**: The fine-tuning of pre-trained language models has a great success in many NLP fields. Yet, it is strikingly vulnerable to adversarial examples, e.g., word substitution attacks using only synonyms can easily fool a BERT-based sentiment analysis model. In this paper, we demonstrate that adversarial training, the prevalent defense technique, does not directly fit a conventional fine-tuning scenario, because it suffers severely from catastrophic forgetting: failing to retain the generic and robust linguistic features that have already been captured by the pre-trained model. In this light, we propose Robust Informative Fine-Tuning (RIFT), a novel adversarial fine-tuning method from an information-theoretical perspective. In particular, RIFT encourages an objective model to retain the features learned from the pre-trained model throughout the entire fine-tuning process, whereas a conventional one only uses the pre-trained weights for initialization. Experimental results show that RIFT consistently outperforms the state-of-the-arts on two popular NLP tasks: sentiment analysis and natural language inference, under different attacks across various pre-trained language models.



## **46. Collaborative adversary nodes learning on the logs of IoT devices in an IoT network**

cs.CR

**SubmitDate**: 2021-12-22    [paper-pdf](http://arxiv.org/pdf/2112.12546v1)

**Authors**: Sandhya Aneja, Melanie Ang Xuan En, Nagender Aneja

**Abstracts**: Artificial Intelligence (AI) development has encouraged many new research areas, including AI-enabled Internet of Things (IoT) network. AI analytics and intelligent paradigms greatly improve learning efficiency and accuracy. Applying these learning paradigms to network scenarios provide technical advantages of new networking solutions. In this paper, we propose an improved approach for IoT security from data perspective. The network traffic of IoT devices can be analyzed using AI techniques. The Adversary Learning (AdLIoTLog) model is proposed using Recurrent Neural Network (RNN) with attention mechanism on sequences of network events in the network traffic. We define network events as a sequence of the time series packets of protocols captured in the log. We have considered different packets TCP packets, UDP packets, and HTTP packets in the network log to make the algorithm robust. The distributed IoT devices can collaborate to cripple our world which is extending to Internet of Intelligence. The time series packets are converted into structured data by removing noise and adding timestamps. The resulting data set is trained by RNN and can detect the node pairs collaborating with each other. We used the BLEU score to evaluate the model performance. Our results show that the predicting performance of the AdLIoTLog model trained by our method degrades by 3-4% in the presence of attack in comparison to the scenario when the network is not under attack. AdLIoTLog can detect adversaries because when adversaries are present the model gets duped by the collaborative events and therefore predicts the next event with a biased event rather than a benign event. We conclude that AI can provision ubiquitous learning for the new generation of Internet of Things.



## **47. Reevaluating Adversarial Examples in Natural Language**

cs.CL

15 pages; 9 Tables; 5 Figures

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2004.14174v3)

**Authors**: John X. Morris, Eli Lifland, Jack Lanchantin, Yangfeng Ji, Yanjun Qi

**Abstracts**: State-of-the-art attacks on NLP models lack a shared definition of a what constitutes a successful attack. We distill ideas from past work into a unified framework: a successful natural language adversarial example is a perturbation that fools the model and follows some linguistic constraints. We then analyze the outputs of two state-of-the-art synonym substitution attacks. We find that their perturbations often do not preserve semantics, and 38% introduce grammatical errors. Human surveys reveal that to successfully preserve semantics, we need to significantly increase the minimum cosine similarities between the embeddings of swapped words and between the sentence encodings of original and perturbed sentences.With constraints adjusted to better preserve semantics and grammaticality, the attack success rate drops by over 70 percentage points.



## **48. MIA-Former: Efficient and Robust Vision Transformers via Multi-grained Input-Adaptation**

cs.CV

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11542v1)

**Authors**: Zhongzhi Yu, Yonggan Fu, Sicheng Li, Chaojian Li, Yingyan Lin

**Abstracts**: ViTs are often too computationally expensive to be fitted onto real-world resource-constrained devices, due to (1) their quadratically increased complexity with the number of input tokens and (2) their overparameterized self-attention heads and model depth. In parallel, different images are of varied complexity and their different regions can contain various levels of visual information, indicating that treating all regions/tokens equally in terms of model complexity is unnecessary while such opportunities for trimming down ViTs' complexity have not been fully explored. To this end, we propose a Multi-grained Input-adaptive Vision Transformer framework dubbed MIA-Former that can input-adaptively adjust the structure of ViTs at three coarse-to-fine-grained granularities (i.e., model depth and the number of model heads/tokens). In particular, our MIA-Former adopts a low-cost network trained with a hybrid supervised and reinforcement training method to skip unnecessary layers, heads, and tokens in an input adaptive manner, reducing the overall computational cost. Furthermore, an interesting side effect of our MIA-Former is that its resulting ViTs are naturally equipped with improved robustness against adversarial attacks over their static counterparts, because MIA-Former's multi-grained dynamic control improves the model diversity similar to the effect of ensemble and thus increases the difficulty of adversarial attacks against all its sub-models. Extensive experiments and ablation studies validate that the proposed MIA-Former framework can effectively allocate computation budgets adaptive to the difficulty of input images meanwhile increase robustness, achieving state-of-the-art (SOTA) accuracy-efficiency trade-offs, e.g., 20% computation savings with the same or even a higher accuracy compared with SOTA dynamic transformer models.



## **49. Improving Robustness with Image Filtering**

cs.CV

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2112.11235v1)

**Authors**: Matteo Terzi, Mattia Carletti, Gian Antonio Susto

**Abstracts**: Adversarial robustness is one of the most challenging problems in Deep Learning and Computer Vision research. All the state-of-the-art techniques require a time-consuming procedure that creates cleverly perturbed images. Due to its cost, many solutions have been proposed to avoid Adversarial Training. However, all these attempts proved ineffective as the attacker manages to exploit spurious correlations among pixels to trigger brittle features implicitly learned by the model. This paper first introduces a new image filtering scheme called Image-Graph Extractor (IGE) that extracts the fundamental nodes of an image and their connections through a graph structure. By leveraging the IGE representation, we build a new defense method, Filtering As a Defense, that does not allow the attacker to entangle pixels to create malicious patterns. Moreover, we show that data augmentation with filtered images effectively improves the model's robustness to data corruption. We validate our techniques on CIFAR-10, CIFAR-100, and ImageNet.



## **50. Adversarial images for the primate brain**

q-bio.NC

These results reveal limits of CNN-based models of primate vision  through their differential response to adversarial attack, and provide clues  for building better models of the brain and more robust computer vision  algorithms

**SubmitDate**: 2021-12-21    [paper-pdf](http://arxiv.org/pdf/2011.05623v2)

**Authors**: Li Yuan, Will Xiao, Gabriel Kreiman, Francis E. H. Tay, Jiashi Feng, Margaret S. Livingstone

**Abstracts**: Convolutional neural networks (CNNs) are vulnerable to adversarial attack, the phenomenon that adding minuscule noise to an image can fool CNNs into misclassifying it. Because this noise is nearly imperceptible to human viewers, biological vision is assumed to be robust to adversarial attack. Despite this apparent difference in robustness, CNNs are currently the best models of biological vision, revealing a gap in explaining how the brain responds to adversarial images. Indeed, sensitivity to adversarial attack has not been measured for biological vision under normal conditions, nor have attack methods been specifically designed to affect biological vision. We studied the effects of adversarial attack on primate vision, measuring both monkey neuronal responses and human behavior. Adversarial images were created by modifying images from one category(such as human faces) to look like a target category(such as monkey faces), while limiting pixel value change. We tested three attack directions via several attack methods, including directly using CNN adversarial images and using a CNN-based predictive model to guide monkey visual neuron responses. We considered a wide range of image change magnitudes that covered attack success rates up to>90%. We found that adversarial images designed for CNNs were ineffective in attacking primate vision. Even when considering the best attack method, primate vision was more robust to adversarial attack than an ensemble of CNNs, requiring over 100-fold larger image change to attack successfully. The success of individual attack methods and images was correlated between monkey neurons and human behavior, but was less correlated between either and CNN categorization. Consistently, CNN-based models of neurons, when trained on natural images, did not generalize to explain neuronal responses to adversarial images.



